"""Specialist callbacks using the existing baseline validation and tool contracts."""
import json
import re

from google.genai import types

from ..single_agent import RESULT_KEY, reset_turn
from ..tools.agent_tools import KEY, email_draft, remember, render_result
from ..tools._records import normalized, response

REQUEST_KEY = "temp:multi_request"


from ..request_context import request_details, user_facts


def _start(callback_context, role):
    callback_context.state["temp:active_role"] = role
    request = callback_context.state.get(REQUEST_KEY)
    if not isinstance(request, dict):
        request = request_details(callback_context.user_content, callback_context.state)
        callback_context.state[REQUEST_KEY] = request
    callback_context.state["requested_semester"] = request["semester"]
    callback_context.state["temp:allow_email_placeholders"] = role == "email"
    # Direct slot handling remains available to the Schedule Agent only.
    direct_slots = role == "schedule" and request.get("agents", [role]) == ["schedule"]
    content = reset_turn(callback_context, route_slots=direct_slots)
    if content is not None:
        return content
    if role == "email" and not request["email_type"]:
        result = {"status": "requires_clarification",
                  "answer": "Should I draft a sick-leave request or a makeup-examination request?",
                  "citations": [], "validation": "abstained"}
        callback_context.state[RESULT_KEY] = result
        return types.Content(role="model", parts=[types.Part(text=result["answer"])])
    if role == "email" and (request["course_code"] or not request["semester"]):
        # An explicitly generic draft is useful even before the student supplies
        # a course/semester; there is no instructor lookup across unknown semesters.
        selected = email_draft(request["course_code"], request["semester"],
                               request["email_type"], callback_context, **user_facts(request["question"]))
        result = render_result(selected["result_id"], callback_context.state[KEY])
        callback_context.state[RESULT_KEY] = result
        return types.Content(role="model", parts=[types.Part(text=result["answer"])])


def start_policy(callback_context):
    return _start(callback_context, "policy")


def start_schedule(callback_context):
    return _start(callback_context, "schedule")


def start_email(callback_context):
    return _start(callback_context, "email")


def mentioned(value, question):
    return isinstance(value, str) and bool(value.strip()) and bool(re.search(
        r"(?<!\w)" + re.escape(normalized(value)) + r"(?!\w)", normalized(question)))


def guard_tool_arguments(tool, args, tool_context):
    """Constrain student facts and semester context before calling existing tools."""
    request = tool_context.state.get(REQUEST_KEY, {})
    question = request.get("question", "")
    name = tool.name
    supplied = dict(args)
    tool_context.state["temp:argument_trace"] = list(tool_context.state.get("temp:argument_trace", [])) + [
        {"tool": name, "supplied": supplied, "user_context": dict(request)}]
    if name == "retrieve_policy":
        args["academic_program"] = request.get("program", "")
        args["admission_year"] = request.get("admission_year", 0)
        return None
    if "semester" in args or name in {
            "course_slot", "courses_in_slot", "elective_count", "instructor",
            "email_draft", "course_lookup", "semester_offerings"}:
        args["semester"] = request.get("semester", "")
    if name in {"course_slot", "course_lookup"} and not mentioned(args.get("course"), question):
        return remember(tool_context, name, response(
            "ambiguous", message="Please specify the course title or code."))
    if name == "courses_in_slot" and not mentioned(args.get("slot"), question):
        return remember(tool_context, name, response(
            "ambiguous", message="Please specify the slot."))
    if name == "elective_count":
        if not mentioned(args.get("department"), question) or not mentioned(str(args.get("level", "")), question):
            return remember(tool_context, name, response(
                "ambiguous", message="Please specify the department and course level."))
    if name == "semester_offerings" and args.get("department") and not mentioned(args["department"], question):
        return remember(tool_context, name, response(
            "ambiguous", message="Please specify the department."))
    if name in {"instructor", "email_draft"}:
        code = args.get("course_code", "")
        resolved = {
            item["data"]["course_code"].upper()
            for item in tool_context.state.get(KEY, {}).values()
            if item.get("tool") == "course_lookup" and item.get("status") == "success"
        }
        known = isinstance(code, str) and (mentioned(code, question) or code.upper() in resolved)
        if not known:
            if name == "instructor":
                return remember(tool_context, name, response(
                    "ambiguous", message="Please specify or resolve the course code first."))
            args["course_code"] = ""
    if name == "semester_offerings":
        args["department"] = "" if not re.search(r"(?:department|departmental)\s+(?:of\s+)?", question, re.I) else args.get("department", "")
    if name == "email_draft":
        args["request_type"] = request.get("email_type", "")
        for field in ("student_name", "roll_number", "start_date", "end_date", "reason"):
            if not mentioned(args.get(field, ""), question):
                args[field] = ""
    tool_context.state["temp:effective_arguments"] = dict(args)
    trace = list(tool_context.state.get("temp:argument_trace", []))
    if trace:
        trace[-1]["effective"] = dict(args)
        tool_context.state["temp:argument_trace"] = trace
    return None


def trace_policy_tool(tool, args, tool_context, tool_response):
    """Make RAG invocation observable without changing retrieval or its payload."""
    if tool.name == "retrieve_policy":
        trace = list(tool_context.state.get("temp:tool_calls", []))
        trace.append({"tool": tool.name, "lookup_status": tool_response.get("status", "unknown"),
                      "academic_program": args.get("academic_program", ""),
                      "admission_year": args.get("admission_year", 0)})
        tool_context.state["temp:tool_calls"] = trace


def project_specialist_task(callback_context, llm_request):
    """Project only the user task, preserving tool evidence and original fact guards."""
    request = callback_context.state.get(REQUEST_KEY, {})
    if len(request.get("agents", [])) <= 1 or callback_context.state.get("temp:active_role") != "policy":
        return None
    question = re.split(r"\b(?:and|then|also)\s+(?:draft|write|compose|prepare)\b",
                        request["question"], maxsplit=1, flags=re.I)[0].strip(" .,")
    contents = []
    for content in llm_request.contents:
        cloned = content.model_copy(deep=True)
        if cloned.role == "user":
            for part in cloned.parts or []:
                if part.text and "Student context:" in part.text and "\nQuestion:" in part.text:
                    header = part.text.split("\nQuestion:", 1)[0]
                    part.text = header + "\nQuestion: " + question
        contents.append(cloned)
    llm_request.contents = contents
    return None
