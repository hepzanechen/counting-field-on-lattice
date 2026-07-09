"""Intake stage: LLM proposes a structured hypothesis; deterministic gates validate.

Power boundary: the LLM picks model/params/criteria, but
- model must exist in CATALOG (gate 1)
- every parameter must pass bounds validation (gate 2)
- criteria may only reference measurable OBSERVABLES (gate 3)
The LLM cannot invent observables, exceed bounds, or skip the control experiment.
"""
from pathlib import Path

from agent.hypothesis import Hypothesis, FalsificationCriterion
from agent.llm import ask_json
from agent.model_catalog import CATALOG, OBSERVABLES, catalog_for_prompt

PROMPT_PATH = Path(__file__).parent / "prompts" / "intake.md"

REQUIRED_KEYS = ["conjecture_restated", "model", "positive_experiment",
                 "control_experiment", "criteria_positive", "criteria_control"]


def _gate_criteria(raw: list, side: str) -> list[FalsificationCriterion]:
    if not raw:
        raise ValueError(f"LLM returned no {side} criteria - not falsifiable")
    gated = []
    for c in raw:
        if c["name"] not in OBSERVABLES:
            raise ValueError(
                f"criterion references unknown observable '{c['name']}' "
                f"(allowed: {OBSERVABLES})")
        if c["comparator"] not in ("<", ">"):
            raise ValueError(f"invalid comparator {c['comparator']!r}")
        gated.append(FalsificationCriterion(
            name=c["name"], description=str(c.get("description", "")),
            threshold=float(c["threshold"]), comparator=c["comparator"]))
    return gated


def intake(conjecture: str) -> tuple[Hypothesis, list[FalsificationCriterion], list[dict]]:
    prompt = PROMPT_PATH.read_text().format(
        catalog=catalog_for_prompt(), conjecture=conjecture)
    proposal = ask_json(prompt, required_keys=REQUIRED_KEYS)

    model_name = proposal["model"]
    if model_name not in CATALOG:
        raise ValueError(f"LLM proposed unknown model {model_name!r} "
                         f"(allowed: {list(CATALOG)})")
    spec = CATALOG[model_name]

    experiments = []
    for side in ("positive_experiment", "control_experiment"):
        exp = proposal[side]
        clean_params = spec.validate(exp["params"])
        experiments.append({
            "label": str(exp["label"]),
            "side": "positive" if side == "positive_experiment" else "control",
            "params": clean_params,
            "rationale": str(exp.get("rationale", "")),
        })

    criteria_positive = _gate_criteria(proposal["criteria_positive"], "positive")
    criteria_control = _gate_criteria(proposal["criteria_control"], "control")

    hypothesis = Hypothesis(
        conjecture=str(proposal["conjecture_restated"]),
        model=model_name,
        parameters={"source": "llm_intake_v1"},
        criteria=criteria_positive)

    return hypothesis, criteria_control, experiments
