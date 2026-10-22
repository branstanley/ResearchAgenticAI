from collections.abc import Callable
from typing import Any

import aisuite as ai
from aisuite import agent_tool


def build_editor() -> tuple[ai.Agent, Callable[..., Any]]:
    agent = ai.Agent(
        name="Research editor",
        model="ollama:qwen3:8b",
        instructions=(
            """
            You are an editor specialized in editing research papers for clarity and correctness. 
            You are to review the provided report and do an editorial pass to ensure the report is clear
            and well written.
            Ensure all sources are cited.
            Return a well formated, clear, error free report with no additional commentary
            """
        ),
        tools=[],
    )
    delegate = agent_tool(
        agent,
        name="delegate_editor",
        description=(
            """
            Delegate research editor, this agent will review a completed report for correctness and conciseness.
            This agent is specialized in taking an already generated report and editing it.  Will ensure sources are cited.
            """
        ),
    )
    return agent, delegate
