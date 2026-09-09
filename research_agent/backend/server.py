"""
backend/server.py – FastAPI backend for the Research Agent frontend.

Provides:
  POST /chat         → Proxy to watsonx Orchestrate agent (or direct watsonx API)
  GET  /health       → Health check
  GET  /pipeline     → Run the full research pipeline directly via watsonx API

Environment variables:
  WATSONX_API_KEY    – IBM Cloud API key (required)
  AGENT_BACKEND_URL  – Optional: watsonx Orchestrate agent endpoint
  CORS_ORIGINS       – Comma-separated allowed origins (default: *)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load from backend/.env first, then fall back to project-root research_agent/.env
_here = Path(__file__).parent
load_dotenv(_here / ".env")
load_dotenv(_here.parent / ".env")

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Research Agent Backend",
    description="FastAPI backend proxying requests to IBM watsonx Granite for the Research Agent",
    version="1.0.0",
)

cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────

WATSONX_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
WATSONX_MODEL = "ibm/granite-4-h-small"
WATSONX_PROJECT_ID = "cbf265ef-b635-441e-9fb3-600fe79d80af"

# ── Schemas ───────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    context: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    thread_id: Optional[str] = None
    model: str
    latency_ms: int


class PipelineRequest(BaseModel):
    topic: str
    research_question: str
    max_papers: int = 8
    source: str = "semantic_scholar"
    word_limit: int = 1500
    citation_format: str = "apa"


class PipelineResponse(BaseModel):
    topic: str
    report: str
    references: str
    papers_found: int
    latency_ms: int
    error: Optional[str] = None


# ── Watsonx helpers ───────────────────────────────────────────────────────────

_token_cache: dict = {}


def _get_iam_token(api_key: str) -> str:
    """Cached IBM IAM token exchange (refreshes when within 2 min of expiry)."""
    now = time.time()
    cached = _token_cache.get(api_key)
    if cached and now < cached["expires_at"] - 120:
        return cached["token"]

    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data["access_token"]
    expires_in = int(data.get("expires_in", 3600))
    _token_cache[api_key] = {"token": token, "expires_at": now + expires_in}
    return token


def _call_watsonx(prompt: str, max_tokens: int = 1000) -> str:
    api_key = os.environ.get("WATSONX_API_KEY", "")
    if not api_key:
        raise ValueError("WATSONX_API_KEY environment variable is not set.")

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


def _search_semantic_scholar(query: str, limit: int) -> list:
    params = {
        "query": query, "limit": min(limit, 20),
        "fields": "title,authors,year,abstract,externalIds,citationCount,url",
    }
    resp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params=params, timeout=15,
    )
    resp.raise_for_status()
    papers = []
    for item in resp.json().get("data", []):
        authors = ", ".join(a.get("name", "") for a in item.get("authors", []))
        papers.append({
            "title": item.get("title", "Unknown"),
            "authors": authors or "Unknown",
            "year": item.get("year"),
            "abstract": item.get("abstract") or "",
            "url": item.get("url") or "",
            "doi": (item.get("externalIds") or {}).get("DOI"),
        })
    return papers


def _search_arxiv(query: str, limit: int) -> list:
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
        url = next(
            (l.get("href", "") for l in entry.findall("atom:link", ns)
             if l.get("rel") == "alternate"), ""
        )
        authors = ", ".join(
            (a.findtext("atom:name", namespaces=ns) or "")
            for a in entry.findall("atom:author", ns)
        )
        pub = entry.findtext("atom:published", namespaces=ns) or ""
        year = int(pub[:4]) if pub else None
        papers.append({"title": title, "authors": authors or "Unknown",
                       "year": year, "abstract": abstract, "url": url})
    return papers


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Health check endpoint."""
    api_key = os.environ.get("WATSONX_API_KEY", "")
    return {
        "status": "ok",
        "model": WATSONX_MODEL,
        "project_id": WATSONX_PROJECT_ID,
        "watsonx_api_key_set": bool(api_key),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint. Sends the user message to watsonx Granite with a
    research assistant system prompt and returns the response.
    """
    t0 = time.time()

    context_block = f"\nContext:\n{req.context}\n" if req.context else ""
    prompt = (
        "You are an expert academic research assistant powered by IBM watsonx Granite. "
        "Answer research questions, help search literature, summarize papers, "
        "organize references, and draft research reports. "
        "Respond in Markdown format with proper headings and citations."
        f"{context_block}\n\n"
        f"User: {req.message}\n\nAssistant:"
    )

    try:
        answer = _call_watsonx(prompt, max_tokens=1200)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except requests.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"watsonx API error: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    latency = int((time.time() - t0) * 1000)
    return ChatResponse(
        response=answer,
        thread_id=req.thread_id or f"thread_{int(time.time())}",
        model=WATSONX_MODEL,
        latency_ms=latency,
    )


@app.post("/pipeline", response_model=PipelineResponse)
def run_pipeline(req: PipelineRequest):
    """
    Run the full research pipeline: search → summarize → draft report.
    Returns the Markdown report and formatted references.
    """
    t0 = time.time()

    # 1 – Search
    try:
        if req.source == "arxiv":
            papers = _search_arxiv(req.topic, req.max_papers)
        else:
            papers = _search_semantic_scholar(req.topic, req.max_papers)
    except Exception as exc:
        return PipelineResponse(
            topic=req.topic, report="", references="",
            papers_found=0, latency_ms=0, error=f"Search failed: {exc}",
        )

    if not papers:
        return PipelineResponse(
            topic=req.topic,
            report="No papers found for the given topic. Try a different search query.",
            references="", papers_found=0, latency_ms=0,
        )

    # 2 – Build summaries block
    summaries_block = ""
    for i, p in enumerate(papers, 1):
        summaries_block += (
            f"\n[{i}] **{p['title']}** ({p['authors']}, {p.get('year') or 'n.d.'})\n"
            f"   Abstract: {(p.get('abstract') or '')[:400]}\n"
        )

    # 3 – Format references (APA)
    ref_lines = []
    for i, p in enumerate(papers, 1):
        doi_part = f" https://doi.org/{p['doi']}" if p.get("doi") else (f" {p['url']}" if p.get("url") else "")
        ref_lines.append(
            f"[{i}] {p['authors']} ({p.get('year') or 'n.d.'}). {p['title']}.{doi_part}"
        )
    references = "\n".join(ref_lines)

    # 4 – Draft report via watsonx
    max_tokens = min(int((req.word_limit or 1500) * 1.4), 2000)
    report_prompt = (
        f"You are a senior academic research writer. Draft a comprehensive research report in Markdown.\n\n"
        f"Topic: {req.topic}\n"
        f"Research Question: {req.research_question}\n\n"
        f"Include these sections (## headings): Introduction, Literature Review, "
        f"Key Themes and Findings, Research Gaps and Future Directions, Conclusion, References\n\n"
        f"Paper summaries to synthesize:\n{summaries_block}\n\n"
        f"Write ~{req.word_limit} words in formal academic style. Cite as [Author, Year].\n\nReport:"
    )

    try:
        report = _call_watsonx(report_prompt, max_tokens=max_tokens)
        if "## References" not in report:
            report += "\n\n## References\n\n" + references
    except Exception as exc:
        report = f"Report generation failed: {exc}\n\n## References\n\n{references}"

    latency = int((time.time() - t0) * 1000)
    return PipelineResponse(
        topic=req.topic,
        report=report,
        references=references,
        papers_found=len(papers),
        latency_ms=latency,
    )
