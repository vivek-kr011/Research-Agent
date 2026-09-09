#!/usr/bin/env bash
# setup.sh – Install the IBM watsonx Orchestrate ADK and prepare the environment
# Run this ONCE before import-all.sh

set -euo pipefail

echo "========================================"
echo "  Research Agent – Setup"
echo "========================================"

# 1. Check Python
python3 --version || { echo "ERROR: Python 3.10+ is required."; exit 1; }

# 2. Install ADK
echo "[1/3] Installing IBM watsonx Orchestrate ADK..."
pip install ibm-watsonx-orchestrate --upgrade

# 3. Install backend dependencies
echo "[2/3] Installing backend dependencies..."
pip install -r "$(dirname "$0")/backend/requirements.txt"

# 4. Prompt for API key
echo "[3/3] Environment setup..."
if [ -z "${WATSONX_API_KEY:-}" ]; then
  echo ""
  echo "  WATSONX_API_KEY is not set."
  read -rp "  Enter your IBM Cloud API Key: " api_key
  export WATSONX_API_KEY="$api_key"
  echo "  export WATSONX_API_KEY='$api_key'" >> ~/.bashrc
  echo "  Added WATSONX_API_KEY to ~/.bashrc"
fi

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Activate a watsonx Orchestrate environment:"
echo "     orchestrate env activate <your-env-name>"
echo ""
echo "  2. Import the agent:"
echo "     ./research_agent/import-all.sh"
echo ""
echo "  3. Start the backend:"
echo "     cd research_agent/backend"
echo "     uvicorn server:app --reload --port 8000"
echo ""
echo "  4. Open the frontend:"
echo "     research_agent/frontend/index.html"
echo "========================================"
