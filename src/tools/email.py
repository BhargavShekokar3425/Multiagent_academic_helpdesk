"""Draft editable academic emails. No sending capability."""
from .faculty import lookup_instructor
from ._records import response


def generate_email_draft(course_code: str, semester: str, request_type: str,
                         student_name: str = "", roll_number: str = "",
                         start_date: str = "", end_date: str = "",
                         reason: str = "", *, allow_placeholders: bool = False) -> dict:
    """Draft sick-leave or makeup-exam requests without inventing personal facts."""
    if request_type not in {"sick_leave", "makeup_exam"}:
        return response("error", message="Use sick_leave or makeup_exam.")
    lookup = lookup_instructor(course_code, semester)
    missing = [key for key, value in (("course_code", course_code), ("semester", semester))
               if not value.strip()]
    if lookup["status"] == "ambiguous" and allow_placeholders and missing:
        # Explicit generic draft: do not choose an instructor from another semester.
        lookup = response("not_found", {"instructors": []})
    if lookup["status"] in {"error", "ambiguous"}:
        return lookup
    instructors = lookup["data"].get("instructors", [])
    recipients = list(dict.fromkeys(r["email"] for r in instructors if r["email"]))
    names = list(dict.fromkeys(r["instructor_name"] for r in instructors if r["instructor_name"]))
    greeting = ", ".join(names) if names else "[Instructor name]"
    name, roll = student_name or "[Student name]", roll_number or "[Roll number]"
    start, end = start_date or "[Start/exam date]", end_date or "[End date]"
    why = reason or "[Brief reason supplied by student]"
    display_course = course_code.upper() if course_code.strip() else "[Course code]"
    if request_type == "sick_leave":
        subject = f"Request for sick leave — {display_course}"
        request = f"I request sick leave from {start} to {end} for {display_course}."
    else:
        subject = f"Request for makeup examination — {display_course}"
        request = (f"I request permission to take a makeup examination for {display_course} "
                   f"after missing the examination on {start}.")
    body = (f"Dear {greeting},\n\n{request}\nReason: {why}\n\n"
            "Please let me know the applicable procedure and supporting documents required.\n\n"
            f"Thank you.\n{name}\nRoll number: {roll}")
    return response("success", {
        "to": recipients or ["[Instructor email]"], "subject": subject, "body": body,
        "requires_student_review": True, "sent": False,
        "instructor_lookup_status": lookup["status"],
        "missing_fields": missing,
    }, lookup["evidence"], "Draft only; review placeholders and details before use.")
