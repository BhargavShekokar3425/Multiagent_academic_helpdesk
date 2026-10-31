"""Specialist callbacks using the existing baseline validation and tool contracts."""
import json
import re

from google.genai import types

from ..single_agent import RESULT_KEY, reset_turn
from ..tools.agent_tools import KEY, email_draft, remember, render_result
from ..tools._records import normalized, response

REQUEST_KEY = "temp:multi_request"


def request_details(content, state):
    text = "".join(p.text or "" for p in ((content.parts or []) if content else []))
    supplied = {}
    question = text
    if text.startswith("Student context: ") and "\nQuestion: " in text:
        header, question = text.split("\nQuestion: ", 1)
        try:
            supplied = json.loads(header[len("Student context: "):])
        except ValueError:
            supplied = {}
        if not isinstance(supplied, dict):
            supplied = {}
    semester = state.get("requested_semester") or supplied.get("semester", "")
    semester = semester if isinstance(semester, str) else ""
    if not semester:
        matches = re.findall(r"\b(\d{4})[-–](\d{2}|\d{4})-S([12])\b", question, re.I)
        if not matches:
            matches = [(year, end, "2" if term.upper() in {"II", "2"} else "1")
                       for year, end, term in re.findall(
                           r"\b(\d{4})[-–](\d{2}|\d{4})\s+(?:semester|sem)\s*(II|I|2|1)\b",
                           question, re.I)]
        values = {f"{year}-{end}-S{term}" for year, end, term in matches}
        semester = next(iter(values)) if len(values) == 1 else ""
    codes = set(re.findall(r"\b[A-Z]{2,5}\d{3,4}\b", question, re.I))
    course_code = next(iter(codes)).upper() if len(codes) == 1 else ""
    lower = question.lower()
    if re.search(r"make[ -]?up|missed.{0,60}(?:exam|examination)", lower):
        email_type = "makeup_exam"
    elif re.search(r"sick|illness|\bleave\b", lower):
        email_type = "sick_leave"
    else:
        email_type = ""
    program = supplied.get("academic_program", "")
    year = supplied.get("admission_year", 0)
    return {"question": question, "semester": semester, "course_code": course_code,
            "program": program if isinstance(program, str) else "",
            "admission_year": year if type(year) is int and year >= 0 else 0,
            "email_type": email_type}


def _start(callback_context, role):
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
    if role == "email" and not request["semester"]:
        # An explicitly generic draft is useful even before the student supplies
        # a course/semester; there is no instructor lookup across unknown semesters.
        selected = email_draft(request["course_code"], request["semester"],
                               request["email_type"], callback_context)
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
    if name == "email_draft":
        args["request_type"] = request.get("email_type", "")
        for field in ("student_name", "roll_number", "start_date", "end_date", "reason"):
            if not mentioned(args.get(field, ""), question):
                args[field] = ""
    return None


def trace_policy_tool(tool, args, tool_context, tool_response):
    """Make RAG invocation observable without changing retrieval or its payload."""
    if tool.name == "retrieve_policy":
        trace = list(tool_context.state.get("temp:tool_calls", []))
        trace.append({"tool": tool.name, "lookup_status": tool_response.get("status", "unknown"),
                      "academic_program": args.get("academic_program", ""),
                      "admission_year": args.get("admission_year", 0)})
        tool_context.state["temp:tool_calls"] = trace
