"""
Literature search tool – searches academic databases (arXiv, CrossRef, Semantic Scholar)
and returns a ranked list of papers relevant to a research topic.
"""
from __future__ import annotations

import json
from typing import List, Optional
import requests
from pydantic import BaseModel, Field

from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission


# ─── Schemas ────────────────────────────────────────────────────────────────

class LiteratureSearchInput(BaseModel):
    query: str = Field(..., description="Research topic or keywords to search for")
    max_results: int = Field(default=10, description="Maximum number of papers to return (1-20)")
    source: str = Field(
        default="semantic_scholar",
        description="Database to search: 'semantic_scholar', 'arxiv', or 'crossref'"
    )


class PaperResult(BaseModel):
    title: str = Field(description="Title of the paper")
    authors: str = Field(description="Comma-separated list of authors")
    year: Optional[int] = Field(None, description="Publication year")
    abstract: str = Field(description="Abstract or summary of the paper")
    url: str = Field(description="URL to access the paper")
    doi: Optional[str] = Field(None, description="Digital Object Identifier if available")
    citation_count: Optional[int] = Field(None, description="Number of citations")


class LiteratureSearchOutput(BaseModel):
    query: str = Field(description="The original search query")
    source: str = Field(description="Database that was searched")
    total_found: int = Field(description="Number of papers found")
    papers: List[PaperResult] = Field(description="List of relevant papers")
    error: Optional[str] = Field(None, description="Error message if search failed")


# ─── Helpers ────────────────────────────────────────────────────────────────

def _search_semantic_scholar(query: str, max_results: int) -> List[PaperResult]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 20),
        "fields": "title,authors,year,abstract,externalIds,citationCount,url",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    papers = []
    for item in data.get("data", []):
        authors = ", ".join(a.get("name", "") for a in item.get("authors", []))
        doi = (item.get("externalIds") or {}).get("DOI")
        papers.append(PaperResult(
            title=item.get("title", "Unknown Title"),
            authors=authors or "Unknown",
            year=item.get("year"),
            abstract=item.get("abstract") or "No abstract available.",
            url=item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
            doi=doi,
            citation_count=item.get("citationCount"),
        ))
    return papers


def _search_arxiv(query: str, max_results: int) -> List[PaperResult]:
    import xml.etree.ElementTree as ET
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": min(max_results, 20)}
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
        abstract = (entry.findtext("atom:summary", namespaces=ns) or "").strip()
        paper_url = ""
        for link in entry.findall("atom:link", ns):
            if link.get("rel") == "alternate":
                paper_url = link.get("href", "")
        authors = ", ".join(
            (a.findtext("atom:name", namespaces=ns) or "") for a in entry.findall("atom:author", ns)
        )
        published = entry.findtext("atom:published", namespaces=ns) or ""
        year = int(published[:4]) if published else None
        papers.append(PaperResult(
            title=title, authors=authors or "Unknown",
            year=year, abstract=abstract or "No abstract available.",
            url=paper_url,
        ))
    return papers


def _search_crossref(query: str, max_results: int) -> List[PaperResult]:
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": min(max_results, 20), "select": "title,author,published,abstract,DOI,URL"}
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    items = response.json().get("message", {}).get("items", [])
    papers = []
    for item in items:
        title = (item.get("title") or ["Unknown Title"])[0]
        authors_list = item.get("author", [])
        authors = ", ".join(f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors_list)
        pub = item.get("published", {}).get("date-parts", [[None]])[0]
        year = pub[0] if pub else None
        papers.append(PaperResult(
            title=title, authors=authors or "Unknown",
            year=year,
            abstract=item.get("abstract") or "No abstract available.",
            url=item.get("URL") or f"https://doi.org/{item.get('DOI', '')}",
            doi=item.get("DOI"),
        ))
    return papers


# ─── Tool ───────────────────────────────────────────────────────────────────

@tool(permission=ToolPermission.READ_ONLY)
def search_literature(input: LiteratureSearchInput) -> LiteratureSearchOutput:
    """
    Search academic literature databases for papers related to a research topic.

    Searches Semantic Scholar, arXiv, or CrossRef and returns a ranked list of
    papers with titles, authors, abstracts, DOIs, and citation counts.

    Args:
        input (LiteratureSearchInput): Query, max results, and source database.

    Returns:
        LiteratureSearchOutput: List of matching papers with metadata.
    """
    try:
        source = input.source.lower()
        if source == "arxiv":
            papers = _search_arxiv(input.query, input.max_results)
        elif source == "crossref":
            papers = _search_crossref(input.query, input.max_results)
        else:
            papers = _search_semantic_scholar(input.query, input.max_results)

        return LiteratureSearchOutput(
            query=input.query,
            source=source,
            total_found=len(papers),
            papers=papers,
        )
    except Exception as exc:
        return LiteratureSearchOutput(
            query=input.query,
            source=input.source,
            total_found=0,
            papers=[],
            error=str(exc),
        )
