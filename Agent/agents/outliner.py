from collections.abc import Callable
from typing import Any

import aisuite as ai
from aisuite import agent_tool


def build_outliner() -> tuple[ai.Agent, Callable[..., Any]]:
    agent = ai.Agent(
        name="Research outliner",
        model="ollama:qwen3:8b",
        instructions=(
            """
            You are a research specialist creating the outline of a large scientific report. 
            You are to review the provided research items, distill the information and create a detailed outline for a report
            Return a well formated, clear, outline in json arrray format with no additional commentary.
            Format: [{"section": "Title of Section", "talking_points" : "A list of talking points"}]
            """
        ),
        tools=[],
    )
    delegate = agent_tool(
        agent,
        name="delegate_outliner",
        description=(
            """
            Delegate research outliner, this agent will review provided research material and output an outline for a report to be written.
            """
        ),
    )
    return agent, delegate
