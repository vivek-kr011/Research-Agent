"""
Research report drafting tool – uses IBM watsonx Granite to compose a
structured academic research report from a collection of paper summaries.
"""
from __future__ import annotations

import os
from typing import List, Optional

import requests
from pydantic import BaseModel, Field

from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission


# ─── Schemas ────────────────────────────────────────────────────────────────

class PaperSummaryInput(BaseModel):
    title: str = Field(..., description="Title of the paper")
    authors: Optional[str] = Field(None, description="Authors")
    year: Optional[int] = Field(None, description="Publication year")
    key_contributions: Optional[str] = Field(None, description="Key contributions as text")
    conclusions: Optional[str] = Field(None, description="Main conclusions")
    one_liner: Optional[str] = Field(None, description="One-sentence summary")


class DraftReportInput(BaseModel):
    topic: str = Field(..., description="Main research topic or title of the report")
    research_question: str = Field(
        ..., description="The central research question being addressed"
    )
    paper_summaries: List[PaperSummaryInput] = Field(
        ..., description="List of paper summaries to synthesize into the report"
    )
    report_sections: Optional[List[str]] = Field(
        default=None,
        description="Custom section names. Defaults to: Introduction, Literature Review, Key Themes, Gaps & Future Work, Conclusion"
    )
    target_audience: Optional[str] = Field(
        default="academic researchers",
        description="Intended audience for the report"
    )
    word_limit: Optional[int] = Field(
        default=1500,
        description="Approximate word limit for the report (500-3000)"
    )


class DraftReportOutput(BaseModel):
    topic: str = Field(description="Report topic")
    research_question: str = Field(description="Central research question")
    report_text: str = Field(description="Full drafted research report in Markdown format")
    section_titles: List[str] = Field(description="List of sections in the report")
    paper_count: int = Field(description="Number of papers synthesized")
    word_estimate: int = Field(description="Estimated word count of the report")
    error: Optional[str] = Field(None, description="Error if drafting failed")


# ─── Watsonx helper ─────────────────────────────────────────────────────────

WATSONX_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
WATSONX_MODEL = "ibm/granite-4-h-small"
WATSONX_PROJECT_ID = "cbf265ef-b635-441e-9fb3-600fe79d80af"


def _get_iam_token(api_key: str) -> str:
    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _call_watsonx(prompt: str, api_key: str, max_tokens: int = 1500) -> str:
    token = _get_iam_token(api_key)
    payload = {
        "model_id": WATSONX_MODEL,
        "project_id": WATSONX_PROJECT_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": min(max_tokens, 2000),
            "repetition_penalty": 1.05,
        },
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(WATSONX_URL, json=payload, headers=headers, timeout=90)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return results[0].get("generated_text", "").strip() if results else ""


# ─── Tool ───────────────────────────────────────────────────────────────────

@tool(permission=ToolPermission.READ_WRITE)
def draft_research_report(input: DraftReportInput) -> DraftReportOutput:
    """
    Draft a structured academic research report by synthesizing multiple paper summaries.

    Uses IBM watsonx Granite to generate a Markdown-formatted research report that
    synthesizes findings from multiple papers, identifies themes, and highlights
    research gaps.

    Args:
        input (DraftReportInput): Topic, research question, paper summaries, and preferences.

    Returns:
        DraftReportOutput: Drafted report in Markdown with section titles and word count.
    """
    api_key = os.environ.get("WATSONX_API_KEY", "")
    if not api_key:
        return DraftReportOutput(
            topic=input.topic, research_question=input.research_question,
            report_text="",
            section_titles=[],
            paper_count=len(input.paper_summaries),
            word_estimate=0,
            error="WATSONX_API_KEY environment variable is not set.",
        )

    default_sections = [
        "Introduction",
        "Literature Review",
        "Key Themes and Findings",
        "Research Gaps and Future Directions",
        "Conclusion",
        "References",
    ]
    sections = input.report_sections or default_sections

    # Build paper summaries block
    summaries_text = ""
    for i, p in enumerate(input.paper_summaries, 1):
        summaries_text += f"\n[{i}] **{p.title}** ({p.authors or 'Unknown'}, {p.year or 'n.d.'})\n"
        if p.one_liner:
            summaries_text += f"   Summary: {p.one_liner}\n"
        if p.key_contributions:
            summaries_text += f"   Contributions: {p.key_contributions}\n"
        if p.conclusions:
            summaries_text += f"   Conclusions: {p.conclusions}\n"

    sections_instruction = "\n".join(f"- {s}" for s in sections)
    word_limit = min(max(input.word_limit or 1500, 500), 3000)
    max_tokens = int(word_limit * 1.4)

    prompt = f"""You are a senior academic research writer. Draft a comprehensive research report in Markdown format.

Topic: {input.topic}
Research Question: {input.research_question}
Target Audience: {input.target_audience or 'academic researchers'}
Approximate Word Limit: {word_limit} words

Include the following sections (use ## for section headings):
{sections_instruction}

Use the following paper summaries as your source material:
{summaries_text}

Guidelines:
- Write in formal academic style
- Cite papers using [Author, Year] format inline
- Synthesize findings across papers rather than summarizing one by one
- Identify common themes and contradictions
- Highlight research gaps and suggest future directions
- Use Markdown formatting (## headings, **bold**, bullet lists)
- Keep within the word limit

Draft the complete research report now:
"""

    try:
        report_text = _call_watsonx(prompt, api_key, max_tokens)
        word_estimate = len(report_text.split())

        # Extract section titles found in the report
        found_sections = [
            line.lstrip("#").strip()
            for line in report_text.split("\n")
            if line.startswith("##")
        ]

        return DraftReportOutput(
            topic=input.topic,
            research_question=input.research_question,
            report_text=report_text,
            section_titles=found_sections or sections,
            paper_count=len(input.paper_summaries),
            word_estimate=word_estimate,
        )
    except Exception as exc:
        return DraftReportOutput(
            topic=input.topic,
            research_question=input.research_question,
            report_text="",
            section_titles=[],
            paper_count=len(input.paper_summaries),
            word_estimate=0,
            error=str(exc),
        )
