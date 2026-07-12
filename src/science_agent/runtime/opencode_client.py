"""OpenCode-backed LLM client. The ONLY place the system talks to an LLM.

Design contract (12-factor agents):
- Factor 4: LLM output must parse as JSON matching a schema, else retry then hard-fail.
- Factor 8: callers own control flow; this module only proposes, never executes.
- LLM proposals NEVER bypass the deterministic gates in llm_intake/llm_reporter.
"""
import json
import re
import subprocess
from typing import Any


class LLMError(RuntimeError):
    pass


DEFAULT_AGENT = "Sisyphus - Ultraworker"


def _extract_text(jsonl: str) -> str:
    chunks = []
    for line in jsonl.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part", {})
        if event.get("type") == "text" and part.get("type") == "text":
            chunks.append(part.get("text", ""))
    return "".join(chunks)


def _extract_json_block(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return brace.group(0)
    raise LLMError(f"no JSON object found in LLM output: {text[:300]!r}")


def ask(prompt: str, agent: str = DEFAULT_AGENT, timeout_s: int = 300) -> str:
    proc = subprocess.run(
        ["opencode", "run", "--format", "json", "--agent", agent],
        input=prompt, capture_output=True, text=True, timeout=timeout_s)
    text = _extract_text(proc.stdout)
    if not text:
        raise LLMError(
            f"empty LLM response (exit={proc.returncode}, "
            f"stderr={proc.stderr[-300:]!r})")
    return text


def ask_json(prompt: str, required_keys: list[str],
             agent: str = DEFAULT_AGENT,
             max_retries: int = 2) -> dict[str, Any]:
    last_error = ""
    for attempt in range(max_retries + 1):
        full_prompt = prompt if attempt == 0 else (
            f"{prompt}\n\nYour previous reply was invalid: {last_error}\n"
            f"Reply again with ONLY the JSON object.")
        text = ask(full_prompt, agent=agent)
        try:
            data = json.loads(_extract_json_block(text))
        except (json.JSONDecodeError, LLMError) as exc:
            last_error = str(exc)
            continue
        missing = [k for k in required_keys if k not in data]
        if missing:
            last_error = f"missing required keys: {missing}"
            continue
        return data
    raise LLMError(f"LLM failed schema after {max_retries + 1} attempts: {last_error}")
