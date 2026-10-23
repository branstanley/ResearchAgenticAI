from collections.abc import Callable
from typing import Any

import aisuite as ai
from aisuite import agent_tool

from search_tools import search_arxiv, web_search, wikipedia_lookup


def build_investigator() -> tuple[ai.Agent, Callable[..., Any]]:
    agent = ai.Agent(
        name="Research Investigator",
        model="ollama:qwen3:8b",
        instructions=(
            """
            You are a research specialist focused on uncovering data and cataloging facts. 
            Find as many useful, relevant facts as you can to this topic, be thorough in finding facts
            and ensure there is plenty of useful information to build a report off of.
            Investigate the assigned question using web_search for general web information, 
            search_arxiv for scholarly research, and wikipedia_lookup for Wikipedia-specific 
            facts. If search_arxiv reports that the source is unavailable, continue with
            the other sources and record that arXiv could not be searched; do not imply
            that the search returned no papers. Return ONLY an accurate list of facts about
            the topic and their source as a JSON array
            """
        ),
        tools=[web_search, search_arxiv, wikipedia_lookup],
    )
    delegate = agent_tool(
        agent,
        name="delegate_investigator",
        description=(
            """
            Delegate research report fact investigation to a research specialist who specializes in uncovering data and cataloging facts.
            The agent has access to web_search for general web information, 
            search_arxiv for scholarly research, and wikipedia_lookup for Wikipedia-specific 
            facts.  They will return a list of facts and their source url as JSON.
            """
        ),
    )
    return agent, delegate
