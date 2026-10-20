import re
from datetime import datetime, date
from pathlib import Path

import aisuite as ai
from aisuite import agent_tool
from search_tools import search_arxiv, web_search, wikipedia_lookup


class PerAgentTraceSink:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.stores: dict[str, ai.tracing.JsonlTraceStore] = {}

    def emit(self, event: ai.tracing.TraceEvent) -> None:
        store = self.stores.get(event.agent_name)
        if store is None:
            safe_agent_name = re.sub(r"[^A-Za-z0-9_-]+", "_", event.agent_name).strip("_")
            path = self.directory / f"{self.timestamp}_{safe_agent_name}.jsonl"
            store = ai.tracing.JsonlTraceStore(path)
            self.stores[event.agent_name] = store
        store.append_event(event)


def main() -> None:
    client = ai.Client(
        provider_configs={
            "ollama": {"base_url": "http://127.0.0.1:11434"}
        }
    )
    
    researcher = ai.Agent(
        name="Research specialist",
        model="ollama:qwen3:8b",
        instructions=(
            """
            You are a research specialist specialized in writing clear, concise reports.
            You have access to using web_search for general web information, 
            search_arxiv for scholarly research, and wikipedia_lookup for Wikipedia-specific 
            facts.  You may be given a curated list of facts, if so ensure you use them.
            Ensure you cite all facts in your report.
            You will return a well formated, accurate, and clear report on the given topic.
            """
        ),
        tools=[web_search, search_arxiv, wikipedia_lookup],
        # Add search_tavily to this list when enabling Tavily.
    )
    
    delegate_research = agent_tool(
        researcher,
        name="delegate_research",
        description=(
            """
            Delegate a research report writing agent who specializes in writing reports.
            The agent can be passed a list of curated facts, but also has access to search tools to look up 
            information for fact checking or additional information.  Will cite it's sources.
            """
        ),
    )
    
    investigator = ai.Agent(
        name="Research Investigator",
        model="ollama:qwen3:8b",
        instructions=(
            """
            You are a research specialist focused on uncovering data and cataloging facts. 
            Investigate the assigned question using web_search for general web information, 
            search_arxiv for scholarly research, and wikipedia_lookup for Wikipedia-specific 
            facts. If search_arxiv reports that the source is unavailable, continue with
            the other sources and record that arXiv could not be searched; do not imply
            that the search returned no papers. Return an accurate list of facts about
            the topic and their source as a JSON.
            """
        ),
        tools=[web_search, search_arxiv, wikipedia_lookup],
        # Add search_tavily to this list when enabling Tavily.
    )

    delegate_investigator = agent_tool(
        investigator,
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

    editor = ai.Agent(
        name="Research editor",
        model="ollama:qwen3:8b",
        instructions=(
            """
            You are an editor specialized in editing research papers for clarity and correctness. 
            You are to review the provided report and do an editorial pass to ensure the report is clear
            and well written.
            Ensure all sources are cited.
            You have access to web_search, search_arxiv, wikipedia_lookup tools if required to verify any facts.
            Return a well formated, clear, error free report.
            """
        ),
        tools=[web_search, search_arxiv, wikipedia_lookup],
        # Add search_tavily to this list when enabling Tavily.
    )

    delegate_editor = agent_tool(
        editor,
        name="delegate_editor",
        description=(
            """
            Delegate research editor, this agent will review a completed report for correctness and conciseness.
            This agent is specialized in taking an already generated report and editing it.  Will ensure sources are cited.
            """
        ),
    )

    manager = ai.Agent(
        name="Research Manager",
        model="ollama:qwen3:8b",
        instructions=(
            f"""
            You are the Research Manager. Understand the user's request, 
            break down research questions into focused tasks, and delegate 
            research topics to your specialists.
            You have access to:
            - an research investigator using delegate_investigator
            - a research report writer using delegate_research
            - an editor that specializes in reviewing reports using delegate_editor
            For research questions, delegate to least two distinct agents before answering. 
            Review the specialist's 
            evidence, synthesize a clear answer, cite the supplied source 
            URLs, and call out important limitations or uncertainty. Do not 
            claim to have verified anything that is not in the research 
            report. If the investigator reports arXiv as unavailable, say so
            explicitly and do not describe that as a search with no results.
            The current date is {date.today().strftime('%Y-%m-%d')}.
            Return a well polished, clear report that is easy to digest.
            """
        ),
        tools=[delegate_research, delegate_investigator, delegate_editor],
    )

    question = input("Ask the Research Manager: ").strip()
    if question:
        trace_sink = PerAgentTraceSink(
            Path(__file__).resolve().parent.parent / "traces"
        )
        result = ai.Runner.run_sync(
            manager,
            question,
            client=client,
            max_turns=6,
            trace_sinks=[trace_sink],
        )
        print(result.final_output)


if __name__ == "__main__":
    main()
