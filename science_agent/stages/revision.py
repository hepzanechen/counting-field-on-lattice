"""Hypothesis revision stage backed by the native OpenCode hypothesis-reviser.

The reviser does not erase failures; it proposes a stricter next hypothesis.
Deterministic code decides whether the original evidence was supported,
falsified, or inadmissible before this module is ever called.
"""
import json
from pathlib import Path

from science_agent.runtime.opencode_client import ask_json

PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "hypothesis_reviser.md"

REQUIRED_KEYS = ["revision_needed", "failure_diagnosis", "revised_hypothesis",
                 "recommended_changes", "integrity_note"]


def revise(hypothesis: str, evidence: list[dict[str, object]]) -> dict[str, object]:
    prompt = PROMPT_PATH.read_text().format(
        hypothesis=hypothesis,
        ledger=json.dumps(evidence, indent=2, default=str))
    return ask_json(prompt, required_keys=REQUIRED_KEYS,
                    agent="hypothesis-reviser")
