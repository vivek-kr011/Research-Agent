#!/usr/bin/env bash
# import-all.sh – Import all Research Agent tools and the agent into watsonx Orchestrate
# Usage: ./research_agent/import-all.sh
# Prerequisites: `orchestrate` CLI authenticated, WATSONX_API_KEY set

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "=========================================="
echo "  Research Agent – watsonx Orchestrate Import"
echo "=========================================="
echo ""

# ── 1. Import Python tools ──────────────────────────────────────────────────
echo "[1/3] Importing Python tools..."

for tool_file in \
  search_literature.py \
  summarize_paper.py \
  organize_references.py \
  draft_report.py \
  watsonx_llm.py; do
    echo "  → Importing tool: ${tool_file}"
    orchestrate tools import -k python -f "${SCRIPT_DIR}/tools/${tool_file}"
done

echo ""

# ── 2. Import Flow tool ──────────────────────────────────────────────────────
echo "[2/3] Importing Flow tool..."

for flow_file in research_flow.py; do
    echo "  → Importing flow: ${flow_file}"
    orchestrate tools import -k flow -f "${SCRIPT_DIR}/tools/${flow_file}"
done

echo ""

# ── 3. Import Agent ──────────────────────────────────────────────────────────
echo "[3/3] Importing Agent..."

for agent_file in research_agent.yaml; do
    echo "  → Importing agent: ${agent_file}"
    orchestrate agents import -f "${SCRIPT_DIR}/agents/${agent_file}"
done

echo ""
echo "=========================================="
echo "  Import complete!"
echo ""
echo "  Start the chat UI:"
echo "    orchestrate chat start"
echo ""
echo "  Then select 'research_agent' to begin."
echo "=========================================="
