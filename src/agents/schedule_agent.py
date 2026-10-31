"""Schedule specialist using the existing deterministic structured-data tools."""
from google.adk.agents import Agent

from ..config import build_llm
from ..single_agent import validate_final
from ..tools.agent_tools import (course_slot, courses_in_slot, elective_count,
                                 instructor, course_lookup, semester_offerings)
from .common import start_schedule, guard_tool_arguments

INSTRUCTION = """
Handle only course codes/titles, recorded semester offerings, slots, elective counts
and instructor mappings. Call the available structured tools. Never answer from
pretrained knowledge and never count courses yourself.
Use course_lookup to resolve a title/code independently of timetable availability.
Use semester_offerings for recorded lists; preserve partial-coverage limitations.
Use course_slot or courses_in_slot for meetings, elective_count for Python counts,
and instructor for recorded contacts. Never invent missing semesters or records.
Return JSON only: {"status":"tool_result","result_id":"exact tool result_id"}.
Do not rewrite tool results or supply your own factual answer.
"""


def build_schedule_agent():
    return Agent(
        name="schedule_agent", description="Semester-specific courses, offerings, slots and instructors.",
        model=build_llm(), instruction=INSTRUCTION,
        tools=[course_slot, courses_in_slot, elective_count, instructor, course_lookup, semester_offerings],
        mode="single_turn", disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
        before_agent_callback=start_schedule, before_tool_callback=guard_tool_arguments,
        after_model_callback=validate_final)
