"""Deterministic ADK coordinator; specialists own generation and validation."""
import json
import re
from copy import deepcopy

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.genai import types

from .agents.common import REQUEST_KEY, request_details
from .agents.policy_agent import build_policy_agent
from .agents.schedule_agent import build_schedule_agent
from .agents.email_agent import build_email_agent
from .integration import obvious_out_of_scope, record_unanswered, unanswered_logging_scope
from .single_agent import RESULT_KEY, EVIDENCE_KEY, ANSWER_MODE_KEY
from .tools.agent_tools import KEY, render_result
from .validation.citations import validate_policy_output
from .validation.extractive import verify_extractive_claims


def route_request(question):
    """Return a bounded specialist plan without model-generated tool decisions."""
    q = question.lower()
    if obvious_out_of_scope(question):
        return []
    email = bool(re.search(
        r"(?:draft|write|compose|prepare).{0,80}(?:email|e-mail|request|letter)|"
        r"(?:sick[- ]leave|makeup[- ]exam|make-up[- ]exam).{0,30}(?:email|request)", q))
    policy = bool(re.search(
        r"policy|policies|regulation|procedure|process|curriculum|credit|attendance|"
        r"requirement|rule|duration|weightage|grading|grade|syllabus|course content", q))
    schedule = bool(re.search(
        r"\bslot\b|timetable|schedule|course code|course title|offering|"
        r"courses (?:offered|available)|(?:list|show).{0,25}courses|elective.{0,30}(?:count|how many)|"
        r"how many.{0,60}elective|who (?:teaches|is teaching)|"
        r"(?:instructor|faculty).{0,20}(?:email|address|for)|"
        r"(?:find|look up|lookup).{0,25}(?:instructor|faculty)", q))
    if email:
        # A draft uses its own verified instructor lookup. A mention of the
        # instructor in a draft request does not create a second schedule task.
        schedule = bool(re.search(r"\bslot\b|timetable|offering|how many", q))
    plan = []
    if policy or (not email and not schedule and re.search(
            r"academic|exam|examination|course|semester|leave|thesis|admission", q)):
        plan.append("policy")
    if schedule:
        plan.append("schedule")
    if email:
        plan.append("email")
    return plan


def rejected(message="The specialist did not produce a validated answer."):
    return {"status": "insufficient_evidence", "answer": message,
            "citations": [], "validation": "rejected",
            "failure_type": "specialist_validation_failed"}


def validate_handoff(role, result, evidence, stored, question):
    """Reconstruct answers from validated claims or server-owned tool results."""
    if not isinstance(result, dict):
        return rejected()
    if result.get("validation") == "academic_applicability_abstention":
        return result
    if result.get("validation") == "deterministic_tool_result":
        allowed = ({"email_draft"} if role == "email"
                   else {"course_slot", "courses_in_slot", "elective_count",
                         "instructor", "course_lookup", "semester_offerings"}
                   if role == "schedule" else set())
        for key, item in stored.items():
            canonical = render_result(key, stored)
            if (item.get("tool") in allowed and all(
                    canonical.get(field) == result.get(field)
                    for field in ("status", "tool", "data", "citations"))):
                if result.get("human_review_log"):
                    canonical["human_review_log"] = result["human_review_log"]
                return canonical
        return rejected("No matching specialist tool result was recorded.")
    if result.get("status") != "answer":
        safe = {
            "insufficient_evidence": "I couldn't produce a sufficiently supported answer from the retrieved passages.",
            "requires_clarification": "Please specify your academic program, admission year, and degree track."
                if role == "policy" else "Please specify the course, semester and request type.",
            "out_of_scope": "This helpdesk supports academic questions and academic email drafts.",
            "error": "The specialist service could not complete the request.",
        }
        status = result.get("status")
        if status not in safe:
            return rejected()
        return {**result, "answer": safe[status], "citations": []}
    if role != "policy":
        return rejected()
    try:
        claims = [{"text": claim["text"], "supports": [
            {"chunk_id": cite["chunk_id"], "quote": cite["quote"]}
            for cite in claim["citations"]]} for claim in result["claims"]]
        checked = validate_policy_output(json.dumps(
            {"status": "answer", "claims": claims}), evidence)
        if checked["status"] != "answer" or result.get("support_check", {}).get("supported") is not True:
            return rejected()
        if result.get("validation") == "verbatim_excerpt_checks_passed":
            verdict = verify_extractive_claims(claims, evidence, question)
            if verdict["supported"] is not True:
                return rejected()
        elif result.get("validation") != "citation_quote_and_model_support_checks_passed":
            return rejected()
        return {**result, "answer": checked["answer"], "citations": checked["citations"],
                "claims": checked["claim_evidence"], "warnings": checked["warnings"]}
    except (KeyError, TypeError, ValueError, AttributeError):
        return rejected()


class HelpdeskOrchestrator(BaseAgent):
    """Execute selected ADK sub-agents once each and combine canonical results."""
    extractive: bool = False

    async def _run_async_impl(self, ctx):
        callback = CallbackContext(ctx)
        request = request_details(ctx.user_content, callback.state)
        plan = route_request(request["question"])
        request["agents"] = plan
        callback.state[REQUEST_KEY] = request
        callback.state[ANSWER_MODE_KEY] = "extractive" if self.extractive else "paraphrase"
        callback.state["temp:unanswered_log"] = None
        sections, calls, traces, diagnostics = [], [], [], []
        with unanswered_logging_scope(reuse_existing=True):
            for role in plan:
                child = next(a for a in self.sub_agents if a.name == role + "_agent")
                callback.state[RESULT_KEY] = None
                callback.state[EVIDENCE_KEY] = {}
                callback.state[KEY] = {}
                yield Event(author=self.name, actions=callback.actions)
                try:
                    child_ctx = ctx
                    if len(plan) > 1:
                        if role == "policy":
                            task = re.split(r"\b(?:and|then|also)\s+(?:draft|write|compose|prepare)\b",
                                            request["question"], maxsplit=1, flags=re.I)[0].strip(" .,")
                        else:
                            task = request["question"]
                        # Original facts remain in REQUEST_KEY; only the model task
                        # is narrowed so policy evidence need not substantiate an email.
                        header = json.dumps({"academic_program": request["program"],
                                             "admission_year": request["admission_year"],
                                             "semester": request["semester"]})
                        child_ctx = ctx.model_copy(update={"user_content": types.Content(role="user", parts=[
                            types.Part(text="Student context: " + header + "\nQuestion: " + task)])})
                    async for event in child.run_async(child_ctx):
                        # Keep the function-call/response history and state deltas;
                        # only the coordinator emits the user-facing final text.
                        filtered = event.model_copy(deep=True)
                        if filtered.content:
                            filtered.content.parts = [
                                p for p in (filtered.content.parts or [])
                                if p.function_call or p.function_response]
                            if not filtered.content.parts:
                                filtered.content = None
                        yield filtered
                    callback = CallbackContext(ctx)
                    result = validate_handoff(
                        role, callback.state.get(RESULT_KEY),
                        list(callback.state.get(EVIDENCE_KEY, {}).values()),
                        callback.state.get(KEY, {}), request["question"])
                except Exception as exc:
                    result = {"status": "error", "answer": "The specialist service could not complete the request.",
                              "citations": [], "validation": "service_error",
                              "failure_type": "specialist_error", "error": type(exc).__name__}
                callback = CallbackContext(ctx)
                calls.append({"agent": child.name, "status": result["status"]})
                traces.extend(deepcopy(callback.state.get("temp:tool_calls", [])))
                diagnostics.append({"agent": child.name, "details": result.get("_debug"),
                                    "arguments": deepcopy(callback.state.get("temp:argument_trace", [])),
                                    "handoff_validation": result.get("validation"),
                                    "result_type": result.get("status"),
                                    "raw_final_output": callback.state.get("temp:last_model_output")})
                sections.append({"agent": child.name, **{
                    k: v for k, v in result.items() if not k.startswith("_")}})
            if not plan:
                result = {"status": "out_of_scope",
                          "answer": "I can help with academic policies, courses, schedules and academic email drafts.",
                          "citations": [], "validation": "scope_refusal"}
            else:
                successes = sum(s["status"] == "answer" for s in sections)
                status = ("answer" if successes == len(sections) else "partial_answer" if successes
                          else "error" if any(s["status"] == "error" for s in sections)
                          else "requires_clarification" if any(s["status"] == "requires_clarification" for s in sections)
                          else "out_of_scope" if all(s["status"] == "out_of_scope" for s in sections)
                          else "insufficient_evidence")
                answer = "\n\n".join(
                    ((s["agent"].replace("_agent", "").capitalize() + ":\n") if len(sections) > 1 else "")
                    + s["answer"] for s in sections)
                warnings = list(dict.fromkeys(w for s in sections for w in s.get("warnings", [])))
                result = {"status": status, "answer": answer,
                          "citations": [c for s in sections for c in s.get("citations", [])],
                          "sections": sections, "warnings": warnings,
                          "validation": "validated_specialist_results", "agent_mode": "multi"}
                if len(sections) == 1:
                    for key in ("tool", "lookup_status", "data", "answer_mode",
                                "support_check", "limitation", "revision_attempted", "revision_succeeded"):
                        if key in sections[0]:
                            result[key] = sections[0][key]
            result["_debug"] = {"routing": plan, "agent_calls": calls, "specialist_diagnostics": diagnostics}
            result["_tool_calls"] = traces
            record_unanswered(callback, result, request["question"])
            callback.state[RESULT_KEY] = result
            text = result["answer"] + "".join("\n\n" + w for w in result.get("warnings", []))
            yield Event(author=self.name, content=types.Content(
                role="model", parts=[types.Part(text=text)]), actions=callback.actions)


def build_orchestrator(extractive=False):
    return HelpdeskOrchestrator(
        name="academic_helpdesk_orchestrator", extractive=extractive,
        description="Routes academic requests to validated policy, schedule and email specialists.",
        sub_agents=[build_policy_agent(extractive), build_schedule_agent(), build_email_agent()])
