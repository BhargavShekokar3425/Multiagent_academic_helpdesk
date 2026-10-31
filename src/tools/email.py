"""Draft editable academic emails. No sending capability."""
from .faculty import lookup_instructor
from ._records import response


def generate_email_draft(course_code: str, semester: str, request_type: str,
                         student_name: str = "", roll_number: str = "",
                         start_date: str = "", end_date: str = "",
                         reason: str = "") -> dict:
    """Draft sick-leave or makeup-exam requests without inventing personal facts."""
    if request_type not in {"sick_leave", "makeup_exam"}:
        return response("error", message="Use sick_leave or makeup_exam.")
    lookup = lookup_instructor(course_code, semester)
    if lookup["status"] in {"error", "ambiguous"}:
        return lookup
    instructors = lookup["data"].get("instructors", [])
    recipients = list(dict.fromkeys(r["email"] for r in instructors if r["email"]))
    names = list(dict.fromkeys(r["instructor_name"] for r in instructors if r["instructor_name"]))
    greeting = ", ".join(names) if names else "[Instructor name]"
    name, roll = student_name or "[Student name]", roll_number or "[Roll number]"
    start, end = start_date or "[Start/exam date]", end_date or "[End date]"
    why = reason or "[Brief reason supplied by student]"
    if request_type == "sick_leave":
        subject = f"Request for sick leave — {course_code.upper()}"
        request = f"I request sick leave from {start} to {end} for {course_code.upper()}."
    else:
        subject = f"Request for makeup examination — {course_code.upper()}"
        request = (f"I request permission to take a makeup examination for {course_code.upper()} "
                   f"after missing the examination on {start}.")
    body = (f"Dear {greeting},\n\n{request}\nReason: {why}\n\n"
            "Please let me know the applicable procedure and supporting documents required.\n\n"
            f"Thank you.\n{name}\nRoll number: {roll}")
    return response("success", {
        "to": recipients or ["[Instructor email]"], "subject": subject, "body": body,
        "requires_student_review": True, "sent": False,
        "instructor_lookup_status": lookup["status"],
    }, lookup["evidence"], "Draft only; review placeholders and details before use.")
