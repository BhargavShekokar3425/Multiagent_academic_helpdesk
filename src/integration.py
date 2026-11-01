"""Application-level refusal and unanswered logging."""
import re
from contextlib import contextmanager
from contextvars import ContextVar
from .tools.logging import log_unanswered

# Share one log attempt between callbacks and runner error handling, even when
# ADK callback state is unavailable after a service/session failure.
_request_log = ContextVar("academic_helpdesk_request_log", default=None)


@contextmanager
def unanswered_logging_scope(reuse_existing=False):
    if reuse_existing and _request_log.get() is not None:
        yield
        return
    token = _request_log.set({})
    try:
        yield
    finally:
        _request_log.reset(token)


def obvious_out_of_scope(question):
    """Narrow known refusal cases; not a complete semantic scope classifier."""
    return bool(re.search(
        r"\bstocks?\b|\bweather\b|\bdining options\b|\brestaurants? (?:on|near) campus\b|"
        r"\b(?:reach|travel to|get to) (?:the )?campus from\b", question, re.I))


def record_unanswered(context, result, question):
    if result.get("status") not in {"out_of_scope", "insufficient_evidence", "error", "partial_answer"} or not question.strip():
        return result
    scope = _request_log.get()
    logged = context.state.get("temp:unanswered_log")
    if logged is None:
        logged = result.get("human_review_log")
    if logged is None and scope is not None:
        logged = scope.get("entry")
    if logged is None:
        logged = log_unanswered(
            question, result["status"],
            result.get("reason") or result.get("error") or result.get("answer", "No answer"))
    context.state["temp:unanswered_log"] = logged
    if scope is not None:
        scope["entry"] = logged
    result["human_review_log"] = logged
    return result
