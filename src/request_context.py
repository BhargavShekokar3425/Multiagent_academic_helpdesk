"""Shared request context and user-fact safeguards; no agent or model imports."""
import json
import re
import csv
from pathlib import Path

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


def intent(question):
    q = question.lower()
    email = bool(re.search(r"(?:draft|write|compose|prepare).{0,80}(?:email|e-mail|letter|request)", q))
    policy = bool(re.search(r"policy|regulation|according to|attendance|leave allowance|exam duration|weightage", q))
    return {"email": email, "policy": policy, "mixed": email and policy}

def user_facts(question):
    facts = {}
    for key, pattern in {
        "student_name": r"(?:my name is|student name:)\s*([A-Za-z]+(?: [A-Za-z]+){0,3})",
        "roll_number": r"(?:roll(?: number)?[: ])\s*([A-Za-z0-9-]+)",
        "start_date": r"(?:from|exam(?:ination)? on)\s+(\d{4}-\d{2}-\d{2})",
        "end_date": r"\bto\s+(\d{4}-\d{2}-\d{2})",
    }.items():
        match = re.search(pattern, question, re.I)
        if match:
            facts[key] = match.group(1).strip()
    if re.search(r"\billness\b", question, re.I):
        facts["reason"] = "illness"
    return facts

def unavailable_reason(request):
    q = request["question"]
    if re.search(r"\bmy\b.{0,50}(?:cgpa|class rank)|scholarship.{0,70}(?:deposited|bank account)", q, re.I):
        return "Personal academic or payment records are unavailable in the verified academic sources."
    if re.search(r"exam(?:ination)? date|exact.{0,30}examination date", q, re.I):
        year = request.get("semester", "").split("-S")[0]
        if year:
            path = Path(__file__).resolve().parents[1] / "data/document_manifest.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                years = {r["academic_year"] for r in csv.DictReader(stream)
                         if r.get("type") == "academic_calendar"}
            if year not in years:
                return "No verified academic calendar or timetable covers the requested academic year."
    return None
