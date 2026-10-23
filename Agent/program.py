import json
from typing import Literal, NotRequired, TypedDict

import aisuite as ai
from langgraph.graph import END, START, StateGraph

from agents import build_editor, build_investigator, build_outliner, build_researcher


class OutlineSection(TypedDict):
    section: str
    talking_points: str | list[str]


class ResearchState(TypedDict):
    question: str
    outline: NotRequired[list[OutlineSection]]
    findings: NotRequired[str]
    section_index: NotRequired[int]
    draft: NotRequired[str]
    completed_sections: NotRequired[list[str]]
    final_report: NotRequired[str]


def _run_agent(agent: ai.Agent, prompt: str, client: ai.Client, max_turns: int = 8) -> str:
    result = ai.Runner.run_sync(
        agent,
        prompt,
        client=client,
        max_turns=8,
    )
    if not isinstance(result.final_output, str):
        raise TypeError(
            f"{agent.name} returned {type(result.final_output).__name__}; "
            "expected report text."
        )
    return result.final_output


def _parse_outline(output: str) -> list[OutlineSection]:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError("The outliner must return a valid JSON array.") from exc

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("The outliner must return a non-empty JSON array.")

    sections: list[OutlineSection] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Outline item {index} must be a JSON object.")

        title = item.get("section")
        talking_points = item.get("talking_points")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Outline item {index} needs a non-empty section.")
        if isinstance(talking_points, str):
            if not talking_points.strip():
                raise ValueError(
                    f"Outline item {index} needs non-empty talking_points."
                )
        elif isinstance(talking_points, list):
            if not talking_points or not all(
                isinstance(point, str) and point.strip()
                for point in talking_points
            ):
                raise ValueError(
                    f"Outline item {index} talking_points must be non-empty "
                    "strings."
                )
        else:
            raise ValueError(
                f"Outline item {index} talking_points must be a string "
                "or an array of strings."
            )

        sections.append(
            {"section": title.strip(), "talking_points": talking_points}
        )
    return sections


def build_research_graph(client: ai.Client):
    investigator, _ = build_investigator()
    researcher, _ = build_researcher()
    editor, _ = build_editor()
    outliner, _ = build_outliner()

    def create_outline(state: ResearchState) -> dict[str, list[OutlineSection]]:
        output = _run_agent(outliner, 
            f"""
            Research question:\n{state["question"]}\n\n
            Investigator findings:\n{state['findings']}\n\n
            """
            , client, max_turns=20)
        return {"outline": _parse_outline(output)}

    def investigate(state: ResearchState) -> dict[str, str]:
        return {"findings": _run_agent(investigator, state["question"], client)}

    def prepare_sections(state: ResearchState) -> dict[str, object]:
        if not state["outline"]:
            raise ValueError("Cannot write a report without outline sections.")
        return {"section_index": 0, "completed_sections": []}

    def route_section(
        state: ResearchState,
    ) -> Literal["researcher", "assemble"]:
        if state["section_index"] < len(state["outline"]):
            return "researcher"
        return "assemble"

    def write_section(state: ResearchState) -> dict[str, str]:
        section = state["outline"][state["section_index"]]
        prompt = (
            f"Research question:\n{state['question']}\n\n"
            f"Current report section:\n{json.dumps(section, ensure_ascii=False)}\n\n"
            f"Investigator findings:\n{state['findings']}\n\n"
            "Write only this section of the report, following its talking "
            "points. Cite relevant supplied source URLs. Do not invent facts "
            "or citations; note gaps where evidence is unavailable."
        )
        return {"draft": _run_agent(researcher, prompt, client)}

    def edit_section(state: ResearchState) -> dict[str, str]:
        section = state["outline"][state["section_index"]]
        prompt = (
            f"Report section:\n{json.dumps(section, ensure_ascii=False)}\n\n"
            f"Investigator findings:\n{state['findings']}\n\n"
            f"Section draft:\n{state['draft']}\n\n"
            "Edit this section for clarity, correctness, and concision. "
            "Preserve supported claims and citations; do not add unsupported "
            "facts. Return only the edited section text."
        )
        return {"draft": _run_agent(editor, prompt, client)}

    def save_section(state: ResearchState) -> dict[str, object]:
        section = state["outline"][state["section_index"]]
        completed = [
            *state["completed_sections"],
            f"## {section['section']}\n\n{state['draft']}",
        ]
        return {
            "completed_sections": completed,
            "section_index": state["section_index"] + 1,
        }

    def assemble_report(state: ResearchState) -> dict[str, str]:
        return {"final_report": "\n\n".join(state["completed_sections"])}

    workflow = StateGraph(ResearchState)
    workflow.add_node("outliner", create_outline)
    workflow.add_node("investigator", investigate)
    workflow.add_node("prepare_sections", prepare_sections)
    workflow.add_node("researcher", write_section)
    workflow.add_node("editor", edit_section)
    workflow.add_node("save_section", save_section)
    workflow.add_node("assemble_report", assemble_report)

    workflow.add_edge(START, "investigator")
    workflow.add_edge("investigator", "outliner")
    workflow.add_edge("outliner", "prepare_sections")
    workflow.add_conditional_edges(
        "prepare_sections",
        route_section,
        {"researcher": "researcher", "assemble": "assemble_report"},
    )
    workflow.add_edge("researcher", "editor")
    workflow.add_edge("editor", "save_section")
    workflow.add_conditional_edges(
        "save_section",
        route_section,
        {"researcher": "researcher", "assemble": "assemble_report"},
    )
    workflow.add_edge("assemble_report", END)
    return workflow.compile()


def main() -> None:
    client = ai.Client(
        provider_configs={
            "ollama": {
                "base_url": "http://127.0.0.1:11434",
                "timeout" : 360
            }
        }
    )
    graph = build_research_graph(client)

    question = input("Ask the Research Chain: ").strip()
    if question:
        result = graph.invoke({"question": question})
        print(result["final_report"])
        with open("./report.md", "w", encoding="utf-8") as file:
            file.write(result["final_report"])
        

if __name__ == "__main__":
    main()
