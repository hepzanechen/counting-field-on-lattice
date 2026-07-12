"""Intake stage: LLM proposes a structured hypothesis; deterministic gates validate.

Power boundary: the LLM picks model/params/criteria, but
- model must exist in CATALOG (gate 1)
- every parameter must pass bounds validation (gate 2)
- criteria may only reference measurable OBSERVABLES (gate 3)
The LLM cannot invent observables, exceed bounds, or skip the control experiment.
"""
from pathlib import Path
from typing import Any

from science_agent.core.domain import ScienceDomain
from science_agent.core.hypothesis import Hypothesis, FalsificationCriterion
from science_agent.runtime.opencode_client import ask_json

INTAKE_PROMPT = Path(__file__).parents[1] / "prompts" / "science_intake.md"
DESIGN_PROMPT = Path(__file__).parents[1] / "prompts" / "experiment_design.md"

INTAKE_KEYS = ["conjecture_restated", "model", "signal_description",
               "control_description", "falsification_strategy", "candidate_observables"]
DESIGN_KEYS = ["positive_experiment", "control_experiment",
               "criteria_positive", "criteria_control"]


def _gate_criteria(
    raw: list[dict[str, Any]], side: str, observables: tuple[str, ...]
) -> list[FalsificationCriterion]:
    if not raw:
        raise ValueError(f"LLM returned no {side} criteria - not falsifiable")
    gated = []
    for c in raw:
        if c["name"] not in observables:
            raise ValueError(
                f"criterion references unknown observable '{c['name']}' "
                f"(allowed: {observables})")
        if c["comparator"] not in ("<", ">"):
            raise ValueError(f"invalid comparator {c['comparator']!r}")
        gated.append(FalsificationCriterion(
            name=c["name"], description=str(c.get("description", "")),
            threshold=float(c["threshold"]), comparator=c["comparator"]))
    return gated


def intake(
    conjecture: str, domain: ScienceDomain
) -> tuple[
    Hypothesis,
    list[FalsificationCriterion],
    list[dict[str, Any]],
]:
    intake_prompt = INTAKE_PROMPT.read_text().format(
        catalog=domain.catalog_for_prompt(), conjecture=conjecture)
    intake_proposal = ask_json(
        intake_prompt, required_keys=INTAKE_KEYS, agent="science-intake")

    model_name = intake_proposal["model"]

    for obs in intake_proposal.get("candidate_observables", []):
        if obs not in domain.observables:
            raise ValueError(f"intake proposed unknown observable {obs!r}")

    design_prompt = DESIGN_PROMPT.read_text().format(
        catalog=domain.catalog_for_prompt(), intake=intake_proposal)
    proposal = ask_json(
        design_prompt, required_keys=DESIGN_KEYS, agent="experiment-designer")

    experiments: list[dict[str, Any]] = []
    for side in ("positive_experiment", "control_experiment"):
        exp = proposal[side]
        clean_params = domain.validate_parameters(model_name, exp["params"])
        experiments.append({
            "label": str(exp["label"]),
            "side": "positive" if side == "positive_experiment" else "control",
            "params": clean_params,
            "rationale": str(exp.get("rationale", "")),
        })

    criteria_positive = _gate_criteria(
        proposal["criteria_positive"], "positive", domain.observables)
    criteria_control = _gate_criteria(
        proposal["criteria_control"], "control", domain.observables)

    hypothesis = Hypothesis(
        conjecture=str(intake_proposal["conjecture_restated"]),
        model=model_name,
        parameters={
            "source": "native_opencode_agents_v1",
            "falsification_strategy": str(intake_proposal.get("falsification_strategy", "")),
        },
        criteria=criteria_positive)

    return hypothesis, criteria_control, experiments
