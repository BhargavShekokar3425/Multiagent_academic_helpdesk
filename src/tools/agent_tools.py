"""Store deterministic tool results for source-faithful agent rendering."""
from google.adk.tools.tool_context import ToolContext
from . import schedule, faculty, email

KEY = "temp:structured_results"


def remember(context, name, result):
    stored = dict(context.state.get(KEY, {}))
    key = f"result_{len(stored) + 1}"
    stored[key] = {"tool": name, **result}
    context.state[KEY] = stored
    trace = list(context.state.get("temp:tool_calls", []))
    trace.append({"tool": name, "lookup_status": result["status"], "result_id": key,
                  "course": result.get("data", {}).get("course", ""),
                  "semester": result.get("data", {}).get("semester", "")})
    context.state["temp:tool_calls"] = trace
    return {"result_id": key, **result}


def course_slot(course: str, semester: str, tool_context: ToolContext) -> dict:
    """Look up an exact course code or title in an explicitly specified semester."""
    return remember(tool_context, "course_slot", schedule.get_course_slot(course, semester))


def courses_in_slot(slot: str, semester: str, tool_context: ToolContext) -> dict:
    """List recorded courses/sections in an exact semester slot."""
    return remember(tool_context, "courses_in_slot", schedule.get_courses_in_slot(slot, semester))


def elective_count(department: str, level: int, semester: str, tool_context: ToolContext) -> dict:
    """Count distinct offered elective course codes; Python computes the count."""
    return remember(tool_context, "elective_count",
                    schedule.count_department_electives(department, level, semester))


def instructor(course_code: str, semester: str, tool_context: ToolContext) -> dict:
    """Look up all instructors assigned to a course in the specified semester."""
    return remember(tool_context, "instructor", faculty.lookup_instructor(course_code, semester))


def email_draft(course_code: str, semester: str, request_type: str,
                tool_context: ToolContext, student_name: str = "", roll_number: str = "",
                start_date: str = "", end_date: str = "", reason: str = "") -> dict:
    """Draft sick_leave or makeup_exam email, using placeholders for missing details."""
    return remember(tool_context, "email_draft", email.generate_email_draft(
        course_code, semester, request_type, student_name, roll_number,
        start_date, end_date, reason))


def render_result(result_id, stored):
    item = stored.get(result_id)
    if not item:
        return {"status": "insufficient_evidence", "answer": "No matching tool result was produced this turn.",
                "citations": [], "validation": "rejected"}
    status, data, name = item["status"], item["data"], item["tool"]
    if status != "success":
        return {"status": "requires_clarification" if status == "ambiguous" else
                         "error" if status == "error" else "insufficient_evidence",
                "answer": item["message"] or "No verified matching record is available.",
                "citations": [], "tool": name, "lookup_status": status, "data": data,
                "validation": "deterministic_tool_result"}
    citations = item["evidence"]
    if name == "course_slot":
        answer = f"{data['course_code']} — {data['course_name']} ({data['semester']}):\n"
        answer += "\n".join(
            f"Section {r['section'] or '[unspecified]'}: slot {r['slot']}, "
            f"{r['day']} {r['start_time']}–{r['end_time']}" for r in data["meetings"])
    elif name == "courses_in_slot":
        answer = "\n".join(
            f"{r['course_code']}: section {r['section'] or '[unspecified]'}, slot {r['slot']}"
            for r in data["meetings"])
    elif name == "elective_count":
        if data.get("coverage_verified") is True:
            answer = f"{data['count']} distinct offered courses in {data['semester']}: "
        else:
            answer = f"{data['count']} distinct offered courses recorded in {data['semester']}: "
        answer += ", ".join(data["course_codes"])
        if data.get("coverage_verified") is not True:
            answer += "\nThe dataset is partial; the official total is not established."
    elif name == "instructor":
        answer = "\n".join(f"{r['instructor_name']} — {r['email']}" for r in data["instructors"])
    else:
        answer = ("To: " + ", ".join(data["to"]) + "\nSubject: " + data["subject"]
                  + "\n\n" + data["body"] + "\n\nDraft only. Review before use.")
    refs = list(dict.fromkeys(
        f"{e['document_id']}, source p. {e['page']}" if e.get("page") else
        f"[{e['document_id']} — {e['source_location']}]({e['source_url']})"
        for e in citations))
    if refs:
        answer += "\n\nSources: " + "; ".join(refs)
    return {"status": "answer", "answer": answer, "citations": citations,
            "data": data, "tool": name, "lookup_status": status,
            "validation": "deterministic_tool_result"}
