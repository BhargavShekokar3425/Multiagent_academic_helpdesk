"""Narrow deterministic routing for direct course-slot questions."""
import re


def parse_slot_question(question):
    patterns = [
        r"^(?:what(?: is|'s)|tell me|find|show(?: me)?)\s+(?:the\s+)?slot\s+(?:for|of)\s+(?:the\s+)?(?:course\s+)?(.+?)\s*[?.]?$",
        r"^what\s+slot\s+is\s+(.+?)\s+in\s*[?.]?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, question.strip(), re.I)
        if match:
            course = match.group(1).strip().strip('"“”').rstrip("?.").strip()
            if course:
                return course
    return None


def semester_label(semester):
    match = re.fullmatch(r"(\d{4})[-–](\d{2}|\d{4})-S([12])", semester.strip(), re.I)
    if not match:
        return semester.strip()
    return f"{match[1]}–{match[2]} Semester {'I' if match[3] == '1' else 'II'}"
