"""
Paper summarization tool – calls IBM watsonx Granite to produce structured
summaries of academic papers given their abstracts or full text.
"""
from __future__ import annotations

import os
from typing import List, Optional

import requests
from pydantic import BaseModel, Field

from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission


# ─── Schemas ────────────────────────────────────────────────────────────────

class SummarizePaperInput(BaseModel):
    title: str = Field(..., description="Title of the paper")
    abstract: str = Field(..., description="Abstract or full text of the paper to summarize")
    authors: Optional[str] = Field(None, description="Author(s) of the paper")
    year: Optional[int] = Field(None, description="Publication year")
    focus: Optional[str] = Field(
        None,
        description="Optional focus area for the summary (e.g., 'methodology', 'results', 'limitations')"
    )


class PaperSummary(BaseModel):
    title: str = Field(description="Paper title")
    authors: Optional[str] = Field(None, description="Authors")
    year: Optional[int] = Field(None, description="Publication year")
    one_liner: str = Field(description="One-sentence summary of the paper")
    key_contributions: List[str] = Field(description="List of key contributions or findings")
    methodology: str = Field(description="Brief description of the methodology used")
    conclusions: str = Field(description="Main conclusions and implications")
    relevance_tags: List[str] = Field(description="Keywords/tags describing the research area")
    full_summary: str = Field(description="Comprehensive 2-3 paragraph summary")
    error: Optional[str] = Field(None, description="Error message if summarization failed")


# ─── Watsonx helper ─────────────────────────────────────────────────────────

WATSONX_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
WATSONX_MODEL = "ibm/granite-4-h-small"
WATSONX_PROJECT_ID = "cbf265ef-b635-441e-9fb3-600fe79d80af"


def _get_iam_token(api_key: str) -> str:
    """Exchange IBM API key for a Bearer token."""
    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _call_watsonx(prompt: str, api_key: str) -> str:
    """Send a prompt to watsonx Granite and return the generated text."""
    token = _get_iam_token(api_key)
    payload = {
        "model_id": WATSONX_MODEL,
        "project_id": WATSONX_PROJECT_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 800,
            "repetition_penalty": 1.05,
        },
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(WATSONX_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return results[0].get("generated_text", "").strip() if results else ""


def _parse_summary_response(raw: str, title: str, authors: Optional[str], year: Optional[int]) -> PaperSummary:
    """Parse the LLM output into a structured PaperSummary."""
    lines = [l.strip() for l in raw.split("\n") if l.strip()]

    one_liner = ""
    contributions: List[str] = []
    methodology = ""
    conclusions = ""
    tags: List[str] = []
    full_paragraphs: List[str] = []

    section = None
    for line in lines:
        ll = line.lower()
        if ll.startswith("one-liner:") or ll.startswith("summary:"):
            one_liner = line.split(":", 1)[-1].strip()
            section = "one_liner"
        elif "key contribution" in ll or "contributions:" in ll:
            section = "contributions"
        elif "methodolog" in ll:
            section = "methodology"
        elif "conclusion" in ll:
            section = "conclusions"
        elif "keyword" in ll or "tag" in ll or "relevance" in ll:
            section = "tags"
        elif "full summary" in ll or "detailed summary" in ll:
            section = "full"
        elif line.startswith("-") or line.startswith("•"):
            content = line.lstrip("-•").strip()
            if section == "contributions":
                contributions.append(content)
            elif section == "tags":
                tags.extend([t.strip() for t in content.split(",") if t.strip()])
        else:
            if section == "methodology":
                methodology += " " + line
            elif section == "conclusions":
                conclusions += " " + line
            elif section == "full":
                full_paragraphs.append(line)

    if not one_liner:
        one_liner = lines[0] if lines else "Summary not available."
    if not contributions:
        contributions = ["See full summary for details."]
    if not methodology:
        methodology = "Methodology details not extracted."
    if not conclusions:
        conclusions = "See full summary for conclusions."
    if not tags:
        tags = ["research", "academic"]
    full_summary = " ".join(full_paragraphs) if full_paragraphs else raw[:800]

    return PaperSummary(
        title=title, authors=authors, year=year,
        one_liner=one_liner,
        key_contributions=contributions,
        methodology=methodology.strip(),
        conclusions=conclusions.strip(),
        relevance_tags=tags,
        full_summary=full_summary,
    )


# ─── Tool ───────────────────────────────────────────────────────────────────

@tool(permission=ToolPermission.READ_ONLY)
def summarize_paper(input: SummarizePaperInput) -> PaperSummary:
    """
    Generate a structured summary of an academic paper using IBM watsonx Granite.

    Takes the paper title and abstract (or full text) and produces a structured
    summary including key contributions, methodology, conclusions, and relevance tags.

    Args:
        input (SummarizePaperInput): Paper details including title, abstract, and optional focus area.

    Returns:
        PaperSummary: Structured summary with key contributions, methodology, and conclusions.
    """
    api_key = os.environ.get("WATSONX_API_KEY", "")
    if not api_key:
        return PaperSummary(
            title=input.title, authors=input.authors, year=input.year,
            one_liner="API key not configured.",
            key_contributions=[], methodology="", conclusions="",
            relevance_tags=[], full_summary="",
            error="WATSONX_API_KEY environment variable is not set.",
        )

    focus_instruction = (
        f"\nFocus particularly on: {input.focus}." if input.focus else ""
    )

    prompt = f"""You are a scientific research assistant. Summarize the following academic paper in a structured format.{focus_instruction}

Title: {input.title}
Authors: {input.authors or 'Unknown'}
Year: {input.year or 'Unknown'}

Abstract / Text:
{input.abstract}

Provide your structured summary in the following format:
One-liner: <one sentence summary>
Key Contributions:
- <contribution 1>
- <contribution 2>
- <contribution 3>
Methodology: <brief methodology description>
Conclusions: <main conclusions and implications>
Relevance Tags: <comma-separated keywords>
Full Summary:
<2-3 paragraph comprehensive summary>
"""

    try:
        raw = _call_watsonx(prompt, api_key)
        return _parse_summary_response(raw, input.title, input.authors, input.year)
    except Exception as exc:
        return PaperSummary(
            title=input.title, authors=input.authors, year=input.year,
            one_liner="Error during summarization.",
            key_contributions=[], methodology="", conclusions="",
            relevance_tags=[], full_summary="",
            error=str(exc),
        )
