"""
Research pipeline flow – orchestrates the full autonomous R&D pipeline:
  1. Search literature
  2. Summarize each paper with watsonx Granite
  3. Organize references
  4. Draft the final research report
"""
from __future__ import annotations

import os
from typing import List, Optional

import requests
from pydantic import BaseModel, Field

from ibm_watsonx_orchestrate.flow_builder.flows import Flow, flow, START, END
from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission


# ─── Shared schemas (self-contained per tool isolation rules) ─────────────

class ResearchPipelineInput(BaseModel):
    topic: str = Field(..., description="Research topic to investigate")
    research_question: str = Field(..., description="The specific research question to answer")
    max_papers: int = Field(default=8, description="Maximum number of papers to retrieve (3-15)")
    citation_format: str = Field(default="apa", description="Citation format: apa, mla, ieee, bibtex")
    word_limit: int = Field(default=1500, description="Approximate word limit for the report")
    source: str = Field(default="semantic_scholar", description="Literature source: semantic_scholar, arxiv, crossref")


class PipelinePaper(BaseModel):
    title: str = Field(description="Paper title")
    authors: str = Field(description="Authors")
    year: Optional[int] = Field(None, description="Publication year")
    abstract: str = Field(description="Abstract text")
    url: str = Field(description="Paper URL")
    doi: Optional[str] = Field(None, description="DOI")
    citation_count: Optional[int] = Field(None, description="Citation count")


class PipelinePaperSummary(BaseModel):
    title: str = Field(description="Paper title")
    authors: Optional[str] = Field(None, description="Authors")
    year: Optional[int] = Field(None, description="Year")
    one_liner: str = Field(description="One-sentence summary")
    key_contributions: str = Field(description="Key contributions as text")
    conclusions: str = Field(description="Main conclusions")


class ResearchPipelineOutput(BaseModel):
    topic: str = Field(description="Research topic")
    papers_found: int = Field(description="Number of papers found")
    papers_summarized: int = Field(description="Number of papers summarized")
    report: str = Field(description="Full research report in Markdown")
    references: str = Field(description="Formatted reference list")
    error: Optional[str] = Field(None, description="Error if pipeline failed")


# ─── Constants ───────────────────────────────────────────────────────────────

WATSONX_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
WATSONX_MODEL = "ibm/granite-4-h-small"
WATSONX_PROJECT_ID = "cbf265ef-b635-441e-9fb3-600fe79d80af"


# ─── Internal helpers (self-contained) ───────────────────────────────────────

def _iam_token(api_key: str) -> str:
    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _wx(prompt: str, api_key: str, max_tokens: int = 800) -> str:
    token = _iam_token(api_key)
    payload = {
        "model_id": WATSONX_MODEL,
        "project_id": WATSONX_PROJECT_ID,
        "input": prompt,
        "parameters": {"decoding_method": "greedy", "max_new_tokens": max_tokens, "repetition_penalty": 1.05},
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(WATSONX_URL, json=payload, headers=headers, timeout=90)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return results[0].get("generated_text", "").strip() if results else ""


def _fetch_semantic_scholar(query: str, limit: int) -> List[PipelinePaper]:
    params = {"query": query, "limit": min(limit, 20),
               "fields": "title,authors,year,abstract,externalIds,citationCount,url"}
    resp = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                        params=params, timeout=15)
    resp.raise_for_status()
    papers = []
    for item in resp.json().get("data", []):
        authors = ", ".join(a.get("name", "") for a in item.get("authors", []))
        doi = (item.get("externalIds") or {}).get("DOI")
        papers.append(PipelinePaper(
            title=item.get("title", "Unknown"),
            authors=authors or "Unknown",
            year=item.get("year"),
            abstract=item.get("abstract") or "",
            url=item.get("url") or "",
            doi=doi,
            citation_count=item.get("citationCount"),
        ))
    return papers


def _fetch_arxiv(query: str, limit: int) -> List[PipelinePaper]:
    import xml.etree.ElementTree as ET
    params = {"search_query": f"all:{query}", "start": 0, "max_results": min(limit, 20)}
    resp = requests.get("http://export.arxiv.org/api/query", params=params, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
        abstract = (entry.findtext("atom:summary", namespaces=ns) or "").strip()
        url = next((l.get("href", "") for l in entry.findall("atom:link", ns)
                    if l.get("rel") == "alternate"), "")
        authors = ", ".join((a.findtext("atom:name", namespaces=ns) or "")
                            for a in entry.findall("atom:author", ns))
        pub = entry.findtext("atom:published", namespaces=ns) or ""
        year = int(pub[:4]) if pub else None
        papers.append(PipelinePaper(
            title=title, authors=authors or "Unknown",
            year=year, abstract=abstract, url=url,
        ))
    return papers


def _summarize_paper_wx(paper: PipelinePaper, api_key: str) -> PipelinePaperSummary:
    prompt = f"""Summarize this academic paper in structured format.

Title: {paper.title}
Authors: {paper.authors}
Year: {paper.year or 'Unknown'}
Abstract: {paper.abstract[:1500]}

Format:
One-liner: <one sentence>
Key Contributions:
- <point 1>
- <point 2>
Conclusions: <main conclusions>
"""
    raw = _wx(prompt, api_key, max_tokens=500)
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    one_liner, contribs, conclusions = "", [], ""
    section = None
    for line in lines:
        ll = line.lower()
        if ll.startswith("one-liner:"):
            one_liner = line.split(":", 1)[-1].strip()
        elif "key contribution" in ll:
            section = "c"
        elif "conclusion" in ll:
            section = "conc"
        elif line.startswith("-") and section == "c":
            contribs.append(line.lstrip("- ").strip())
        elif section == "conc" and not line.lower().startswith("conclusion"):
            conclusions += " " + line
    return PipelinePaperSummary(
        title=paper.title, authors=paper.authors, year=paper.year,
        one_liner=one_liner or (lines[0] if lines else "Summary unavailable."),
        key_contributions="; ".join(contribs) if contribs else "See abstract.",
        conclusions=conclusions.strip() or "See abstract for conclusions.",
    )


def _format_apa(papers: List[PipelinePaper]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        venue = ""
        doi_part = f" https://doi.org/{p.doi}" if p.doi else (f" {p.url}" if p.url else "")
        lines.append(f"[{i}] {p.authors} ({p.year or 'n.d.'}). {p.title}.{venue}{doi_part}")
    return "\n".join(lines)


# ─── Pipeline tool (used by the flow) ─────────────────────────────────────

@tool(permission=ToolPermission.READ_WRITE)
def run_research_pipeline(input: ResearchPipelineInput) -> ResearchPipelineOutput:
    """
    Run the complete autonomous R&D research pipeline end-to-end.

    Searches literature, summarizes each paper with IBM watsonx Granite,
    organizes references, and drafts a full research report – all in one call.

    Args:
        input (ResearchPipelineInput): Topic, research question, and preferences.

    Returns:
        ResearchPipelineOutput: Full report, formatted references, and metadata.
    """
    api_key = os.environ.get("WATSONX_API_KEY", "")
    if not api_key:
        return ResearchPipelineOutput(
            topic=input.topic, papers_found=0, papers_summarized=0,
            report="", references="",
            error="WATSONX_API_KEY environment variable is not set.",
        )

    # 1 – Search literature
    try:
        if input.source == "arxiv":
            papers = _fetch_arxiv(input.topic, input.max_papers)
        else:
            papers = _fetch_semantic_scholar(input.topic, input.max_papers)
    except Exception as exc:
        return ResearchPipelineOutput(
            topic=input.topic, papers_found=0, papers_summarized=0,
            report="", references="", error=f"Search failed: {exc}",
        )

    if not papers:
        return ResearchPipelineOutput(
            topic=input.topic, papers_found=0, papers_summarized=0,
            report="No papers found for the given topic.",
            references="",
        )

    # 2 – Summarize papers
    summaries: List[PipelinePaperSummary] = []
    for paper in papers:
        try:
            summaries.append(_summarize_paper_wx(paper, api_key))
        except Exception:
            summaries.append(PipelinePaperSummary(
                title=paper.title, authors=paper.authors, year=paper.year,
                one_liner=paper.abstract[:120] or "Summary unavailable.",
                key_contributions="N/A", conclusions="N/A",
            ))

    # 3 – Format references
    references = _format_apa(papers)

    # 4 – Draft report
    summaries_block = ""
    for i, s in enumerate(summaries, 1):
        summaries_block += (
            f"\n[{i}] **{s.title}** ({s.authors}, {s.year or 'n.d.'})\n"
            f"   Summary: {s.one_liner}\n"
            f"   Contributions: {s.key_contributions}\n"
            f"   Conclusions: {s.conclusions}\n"
        )

    max_tokens = min(int((input.word_limit or 1500) * 1.4), 2000)
    report_prompt = f"""You are a senior academic research writer. Draft a comprehensive research report in Markdown.

Topic: {input.topic}
Research Question: {input.research_question}

Include sections: ## Introduction, ## Literature Review, ## Key Themes and Findings, ## Research Gaps and Future Directions, ## Conclusion, ## References

Paper summaries to synthesize:
{summaries_block}

Write a scholarly, well-structured report (~{input.word_limit} words). Cite papers as [Author, Year].

Report:
"""

    try:
        report_text = _wx(report_prompt, api_key, max_tokens=max_tokens)
        # Append formatted references if not already included
        if "## References" not in report_text and "# References" not in report_text:
            report_text += "\n\n## References\n\n" + references
    except Exception as exc:
        report_text = f"Report generation failed: {exc}\n\n## References\n\n{references}"

    return ResearchPipelineOutput(
        topic=input.topic,
        papers_found=len(papers),
        papers_summarized=len(summaries),
        report=report_text,
        references=references,
    )


# ─── Flow definition ─────────────────────────────────────────────────────────

@flow(
    name="research_pipeline_flow",
    display_name="Research Pipeline Flow",
    description=(
        "Autonomous R&D pipeline: search literature, summarize papers, "
        "organize references, and draft a research report."
    ),
    input_schema=ResearchPipelineInput,
)
def build_research_pipeline_flow(aflow: Flow) -> Flow:
    """
    CRITICAL: Flow function signature is def build_<name>(aflow: Flow) -> Flow:
    """
    pipeline_node = aflow.tool(run_research_pipeline)
    aflow.sequence(START, pipeline_node, END)
    return aflow
