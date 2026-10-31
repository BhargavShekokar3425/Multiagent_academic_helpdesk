"""Application-level refusal and unanswered logging."""
import re
from .tools.logging import log_unanswered


def obvious_out_of_scope(question):
    """Narrow known refusal cases; not a complete semantic scope classifier."""
    return bool(re.search(
        r"\bdining options\b|\brestaurants? (?:on|near) campus\b|"
        r"\b(?:reach|travel to|get to) (?:the )?campus from\b", question, re.I))


def record_unanswered(context, result, question):
    if result.get("status") not in {"out_of_scope", "insufficient_evidence", "error"} or not question.strip():
        return result
    logged = context.state.get("temp:unanswered_log")
    if logged is None:
        logged = log_unanswered(question, result["status"],
                                result.get("reason") or result.get("answer", "No answer"))
        context.state["temp:unanswered_log"] = logged
    result["human_review_log"] = logged
    return result
