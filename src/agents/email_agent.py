"""Draft-only email specialist; instructor addresses come from structured sources."""
from google.adk.agents import Agent

from ..config import build_llm
from ..single_agent import validate_final
from ..tools.agent_tools import email_draft, instructor, course_lookup
from .common import start_email, guard_tool_arguments

INSTRUCTION = """
Handle only the email part of the student's request. Draft a sick_leave or makeup_exam
request using email_draft; use instructor for verified contacts and course_lookup
to resolve a supplied course title. You have no email-sending capability.
Never invent a course, semester, student name, roll number, dates or medical facts.
Copy optional student facts only when explicitly supplied in the original request.
Leave unavailable details empty so the existing tool uses visible placeholders.
If policy explanation is also requested, leave it to the Policy Agent.
Return JSON only: {"status":"tool_result","result_id":"exact tool result_id"}.
Do not rewrite the draft or invent recipients, approvals or policy requirements.
"""


def build_email_agent():
    return Agent(
        name="email_agent", description="Unsent sick-leave/makeup-exam drafts and verified instructor contacts.",
        model=build_llm(), instruction=INSTRUCTION, tools=[email_draft, instructor, course_lookup],
        mode="single_turn", disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
        before_agent_callback=start_email, before_tool_callback=guard_tool_arguments,
        after_model_callback=validate_final)
