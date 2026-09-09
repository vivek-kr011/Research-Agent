"""
Reference organizer tool – stores, deduplicates, formats, and exports
a bibliography from collected paper metadata.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission


# ─── Schemas ────────────────────────────────────────────────────────────────

class ReferenceEntry(BaseModel):
    title: str = Field(..., description="Title of the paper")
    authors: str = Field(..., description="Authors of the paper")
    year: Optional[int] = Field(None, description="Publication year")
    journal_or_venue: Optional[str] = Field(None, description="Journal or conference name")
    doi: Optional[str] = Field(None, description="DOI of the paper")
    url: Optional[str] = Field(None, description="URL to the paper")
    tags: Optional[str] = Field(None, description="Comma-separated topic tags")


class OrganizeReferencesInput(BaseModel):
    references: List[ReferenceEntry] = Field(
        ..., description="List of reference entries to organize"
    )
    format: str = Field(
        default="apa",
        description="Citation format: 'apa', 'mla', 'chicago', 'ieee', or 'bibtex'"
    )
    sort_by: str = Field(
        default="year",
        description="Sort order: 'year', 'author', or 'title'"
    )
    deduplicate: bool = Field(
        default=True,
        description="Remove duplicate references based on title similarity"
    )


class FormattedReference(BaseModel):
    index: int = Field(description="Reference number")
    citation: str = Field(description="Formatted citation string")
    doi: Optional[str] = Field(None, description="DOI if available")
    url: Optional[str] = Field(None, description="URL if available")
    tags: Optional[str] = Field(None, description="Topic tags")


class OrganizeReferencesOutput(BaseModel):
    format: str = Field(description="Citation format used")
    total_references: int = Field(description="Total number of unique references")
    references: List[FormattedReference] = Field(description="Formatted reference list")
    bibtex_block: Optional[str] = Field(None, description="Full BibTeX block (when format='bibtex')")
    duplicates_removed: int = Field(default=0, description="Number of duplicate entries removed")


# ─── Formatting helpers ──────────────────────────────────────────────────────

def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.lower().strip())


def _deduplicate(refs: List[ReferenceEntry]) -> tuple[List[ReferenceEntry], int]:
    seen: set[str] = set()
    unique: List[ReferenceEntry] = []
    for r in refs:
        key = _normalize_title(r.title)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique, len(refs) - len(unique)


def _sort_refs(refs: List[ReferenceEntry], sort_by: str) -> List[ReferenceEntry]:
    if sort_by == "author":
        return sorted(refs, key=lambda r: r.authors.lower())
    elif sort_by == "title":
        return sorted(refs, key=lambda r: r.title.lower())
    else:  # year (default)
        return sorted(refs, key=lambda r: r.year or 0, reverse=True)


def _apa(r: ReferenceEntry, idx: int) -> str:
    year = f"({r.year})" if r.year else "(n.d.)"
    venue = f" *{r.journal_or_venue}*." if r.journal_or_venue else "."
    doi_part = f" https://doi.org/{r.doi}" if r.doi else (f" {r.url}" if r.url else "")
    return f"{r.authors} {year}. {r.title}.{venue}{doi_part}"


def _mla(r: ReferenceEntry, idx: int) -> str:
    year = str(r.year) if r.year else "n.d."
    venue = f" *{r.journal_or_venue}*," if r.journal_or_venue else ","
    doi_part = f" doi:{r.doi}." if r.doi else (f" {r.url}." if r.url else ".")
    return f'{r.authors}. "{r.title}."{venue} {year}.{doi_part}'


def _chicago(r: ReferenceEntry, idx: int) -> str:
    year = str(r.year) if r.year else "n.d."
    venue = f" *{r.journal_or_venue}*." if r.journal_or_venue else "."
    doi_part = f" https://doi.org/{r.doi}" if r.doi else (f" {r.url}" if r.url else "")
    return f'{r.authors}. "{r.title}."{venue} {year}.{doi_part}'


def _ieee(r: ReferenceEntry, idx: int) -> str:
    year = str(r.year) if r.year else "n.d."
    venue = f", *{r.journal_or_venue}*" if r.journal_or_venue else ""
    doi_part = f", doi: {r.doi}" if r.doi else ""
    return f"[{idx}] {r.authors}, \"{r.title}\"{venue}, {year}{doi_part}."


def _bibtex_entry(r: ReferenceEntry, idx: int) -> str:
    first_author = r.authors.split(",")[0].split()[-1] if r.authors else "Unknown"
    key = f"{first_author}{r.year or 'nd'}{idx}"
    entry_type = "article" if r.journal_or_venue else "misc"
    lines = [f"@{entry_type}{{{key},"]
    lines.append(f'  title = {{{r.title}}},')
    lines.append(f'  author = {{{r.authors}}},')
    if r.year:
        lines.append(f'  year = {{{r.year}}},')
    if r.journal_or_venue:
        lines.append(f'  journal = {{{r.journal_or_venue}}},')
    if r.doi:
        lines.append(f'  doi = {{{r.doi}}},')
    if r.url:
        lines.append(f'  url = {{{r.url}}},')
    lines.append("}")
    return "\n".join(lines)


_FORMATTERS = {"apa": _apa, "mla": _mla, "chicago": _chicago, "ieee": _ieee}


# ─── Tool ───────────────────────────────────────────────────────────────────

@tool(permission=ToolPermission.READ_WRITE)
def organize_references(input: OrganizeReferencesInput) -> OrganizeReferencesOutput:
    """
    Organize, deduplicate, sort, and format a list of academic references.

    Accepts raw reference entries and returns them formatted in APA, MLA, Chicago,
    IEEE, or BibTeX style, sorted by year, author, or title, with duplicates removed.

    Args:
        input (OrganizeReferencesInput): References to organize with formatting preferences.

    Returns:
        OrganizeReferencesOutput: Formatted reference list with optional BibTeX block.
    """
    refs = list(input.references)
    removed = 0

    if input.deduplicate:
        refs, removed = _deduplicate(refs)

    refs = _sort_refs(refs, input.sort_by)

    fmt = input.format.lower()
    formatter = _FORMATTERS.get(fmt, _apa)

    formatted: List[FormattedReference] = []
    bibtex_parts: List[str] = []

    for i, ref in enumerate(refs, start=1):
        if fmt == "bibtex":
            citation = _bibtex_entry(ref, i)
            bibtex_parts.append(citation)
        else:
            citation = formatter(ref, i)

        formatted.append(FormattedReference(
            index=i,
            citation=citation if fmt != "bibtex" else _apa(ref, i),
            doi=ref.doi,
            url=ref.url,
            tags=ref.tags,
        ))

    return OrganizeReferencesOutput(
        format=fmt,
        total_references=len(formatted),
        references=formatted,
        bibtex_block="\n\n".join(bibtex_parts) if bibtex_parts else None,
        duplicates_removed=removed,
    )
