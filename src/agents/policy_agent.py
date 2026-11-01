"""Policy specialist: the baseline's RAG, citations, support checks and revision."""
from google.adk.agents import Agent

from ..config import build_llm
from ..single_agent import INSTRUCTION, EXTRACTIVE_INSTRUCTION, retrieve_policy, validate_final
from .common import start_policy, guard_tool_arguments, trace_policy_tool, project_specialist_task


def build_policy_agent(extractive=False):
    return Agent(
        name="policy_agent",
        description="Academic policy, curriculum, credits, attendance and examination evidence.",
        model=build_llm(),
        instruction=INSTRUCTION + """
SPECIALIST ROLE: Answer only the policy/regulation part of the original request.
Use retrieve_policy. Do not draft emails, select recipients or guess timetable data.
When the question also requests an email, leave that part to the Email Agent.
""" + (EXTRACTIVE_INSTRUCTION if extractive else ""),
        tools=[retrieve_policy], mode="single_turn",
        disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
        before_agent_callback=start_policy, before_tool_callback=guard_tool_arguments,
        before_model_callback=project_specialist_task, after_tool_callback=trace_policy_tool, after_model_callback=validate_final)
