"""ADK entry point; default single-agent baseline or explicit multi-agent mode."""
from .config import helpdesk_agent_mode
from .single_agent import baseline_agent


def build_root_agent(mode=None):
    if helpdesk_agent_mode(mode) == "single":
        return baseline_agent
    from .orchestration import build_orchestrator
    return build_orchestrator()


root_agent = build_root_agent()
