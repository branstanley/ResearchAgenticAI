import json
import threading

import arxiv
import wikipedia
from ddgs import DDGS

wikipedia.set_user_agent("ResearchAssistant/0.1 (local research assistant)")

_ARXIV_CLIENT = arxiv.Client(page_size=10, delay_seconds=3.0, num_retries=3)
_ARXIV_LOCK = threading.Lock()


def _validate_search(query: str, max_results: int) -> None:
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10")


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return titles, URLs, and short result descriptions."""
    _validate_search(query, max_results)
    results = DDGS().text(query, max_results=max_results)
    return json.dumps(results, ensure_ascii=False)


def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv papers by topic and return metadata, abstracts, and links."""
    _validate_search(query, max_results)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    try:
        with _ARXIV_LOCK:
            papers = list(_ARXIV_CLIENT.results(search))
    except arxiv.HTTPError as exc:
        if exc.status != 429:
            raise
        return json.dumps(
            {
                "status": "unavailable",
                "source": "arXiv",
                "http_status": exc.status,
                "message": (
                    "arXiv rate-limited this request after retries. Continue "
                    "with other available sources and disclose that arXiv "
                    "could not be searched."
                ),
            },
            ensure_ascii=False,
        )
    return json.dumps(
        [
            {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.published.isoformat(),
                "abstract": paper.summary,
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
            }
            for paper in papers
        ],
        ensure_ascii=False,
    )


def wikipedia_lookup(query: str) -> str:
    """Look up a Wikipedia article and return its title, summary, and URL."""
    if not query.strip():
        raise ValueError("query must not be empty")
    page = wikipedia.page(query, auto_suggest=True, redirect=True)
    summary = wikipedia.summary(
        page.title,
        sentences=5,
        auto_suggest=False,
        redirect=True,
    )
    return f"{page.title}\n{page.url}\n{summary}"


# Enable this optional provider when you are ready:
# 1. Uncomment `tavily-python` in requirements.txt.
# 2. Set the TAVILY_API_KEY environment variable.
# 3. Uncomment the import, tool function, and tool registration below.
#
# import os
# from tavily import TavilyClient
#
#
# def search_tavily(query: str, max_results: int = 5) -> str:
#     """Search the web using Tavily and return sourced results."""
#     _validate_search(query, max_results)
#     client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
#     results = client.search(
#         query=query,
#         max_results=max_results,
#         search_depth="basic",
#     )
#     return json.dumps(results, ensure_ascii=False)
