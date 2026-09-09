"""
main_flow.py – programmatic test runner for the Research Pipeline Flow.

Usage:
    export PYTHONPATH=/path/to/adk/src:/path/to/adk
    export WATSONX_API_KEY=<your-ibm-api-key>
    python3 main_flow.py
"""
import asyncio
from pathlib import Path

from research_agent.tools.research_flow import build_research_pipeline_flow


async def main() -> None:
    flow_def = await build_research_pipeline_flow().compile_deploy()

    generated_folder = Path(__file__).resolve().parent / "research_agent" / "generated"
    generated_folder.mkdir(parents=True, exist_ok=True)

    flow_def.dump_spec(str(generated_folder / "research_pipeline_flow.json"))

    # Sample test invocation
    await flow_def.invoke(
        {
            "topic": "large language models in scientific discovery",
            "research_question": "How are large language models being applied to accelerate scientific research?",
            "max_papers": 5,
            "citation_format": "apa",
            "word_limit": 1000,
            "source": "semantic_scholar",
        },
        debug=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
