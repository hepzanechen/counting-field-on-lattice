"""Skeptical PhD stage: constrained confounder analysis.

The skeptical agent reviews evidence and proposes alternative explanations
ONLY from a known confounders catalog. It does not run experiments or judge
results. Its output is a structured confounder assessment.
"""
import json
from pathlib import Path
from typing import Any

from science_agent.runtime.opencode_client import ask_json

PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "skeptical_phd.md"

REQUIRED_KEYS = ["confounders_considered", "overall_assessment", "recommended_controls"]


def assess(
    evidence: list[dict[str, Any]], verdict: str
) -> dict[str, Any]:
    prompt = PROMPT_PATH.read_text().format(
        ledger=json.dumps(evidence, indent=2, default=str),
        verdict=verdict)
    return ask_json(prompt, required_keys=REQUIRED_KEYS, agent="skeptical-phd")
