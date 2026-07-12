"""Reporting stage: LLM narrates the ledger; deterministic gate verifies citations.

Power boundary: the LLM writes prose, but
- the verdict is computed by code and passed in (LLM cannot overturn it)
- every [E<n>] citation is checked against the ledger (gate 1)
- a report with zero citations is rejected (gate 2)
"""
import json
import re
from pathlib import Path
from typing import Any

from science_agent.runtime.opencode_client import ask

PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "evidence_reporter.md"


def _format_ledger(evidence: list[dict[str, Any]]) -> str:
    lines = []
    for i, entry in enumerate(evidence, start=1):
        lines.append(f"[E{i}] experiment={entry['experiment']} "
                     f"verdict={entry['verdict']}")
        lines.append(f"     measured: {json.dumps(entry['measured'])}")
        failed = [r["invariant"] for r in entry["judge"] if not r["passed"]]
        lines.append(f"     judge: {'all passed' if not failed else f'FAILED {failed}'}")
        for c in entry.get("criteria", []):
            lines.append(f"     criterion {c['criterion']}: value={c['value']:.6g} "
                         f"satisfied={c['satisfied']}")
    return "\n".join(lines)


def report(
    evidence: list[dict[str, Any]], verdict: str, max_retries: int = 1
) -> str:
    ledger_text = _format_ledger(evidence)
    prompt = PROMPT_PATH.read_text().format(ledger=ledger_text, verdict=verdict)

    last_error = ""
    for attempt in range(max_retries + 1):
        full_prompt = prompt if attempt == 0 else (
            f"{prompt}\n\nYour previous report was rejected: {last_error}. "
            f"Write it again following the hard rules.")
        text = ask(full_prompt, agent="evidence-reporter")

        cited = {int(m) for m in re.findall(r"\[E(\d+)\]", text)}
        valid = set(range(1, len(evidence) + 1))
        if not cited:
            last_error = "no [E<n>] citations found"
            continue
        if not cited.issubset(valid):
            last_error = f"cites nonexistent evidence entries {sorted(cited - valid)}"
            continue
        if verdict not in text:
            last_error = f"report must state the computed verdict '{verdict}'"
            continue
        return text

    raise RuntimeError(f"reporter failed citation gate: {last_error}")
