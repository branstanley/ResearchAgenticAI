from collections.abc import Callable
from typing import Any

import aisuite as ai
from aisuite import agent_tool

from search_tools import search_arxiv, web_search, wikipedia_lookup


def build_researcher() -> tuple[ai.Agent, Callable[..., Any]]:
    agent = ai.Agent(
        name="Research specialist",
        model="ollama:qwen3:8b",
        instructions=(
            """
            You are a research specialist specialized in writing clear, concise reports.
            You will be given a topic section with some talking points in the form of a JSON, as well as a list of research facts.
            You will ensure to create a clear, well written section.
            You have access to using web_search for general web information, 
            search_arxiv for scholarly research, and wikipedia_lookup for Wikipedia-specific 
            facts.  You may be given a curated list of facts, if so ensure you use them.
            Ensure you cite all facts in your report.
            You will return a well formated, accurate, and clear report on the given topic with no additional commentary.
            """
        ),
        tools=[web_search, search_arxiv, wikipedia_lookup],
    )
    delegate = agent_tool(
        agent,
        name="delegate_research",
        description=(
            """
            Delegate a research report writing agent who specializes in writing reports.
            The agent can be passed a list of curated facts, but also has access to search tools to look up 
            information for fact checking or additional information.  Will cite it's sources.
            """
        ),
    )
    return agent, delegate
