"""Intake stage: LLM proposes a structured hypothesis; deterministic gates validate.

Power boundary: the LLM picks model/params/criteria, but
- model must exist in CATALOG (gate 1)
- every parameter must pass bounds validation (gate 2)
- criteria may only reference measurable OBSERVABLES (gate 3)
The LLM cannot invent observables, exceed bounds, or skip the control experiment.
"""
from pathlib import Path

from science_agent.core.hypothesis import Hypothesis, FalsificationCriterion
from science_agent.core.model_catalog import CATALOG, OBSERVABLES, catalog_for_prompt
from science_agent.runtime.opencode_client import ask_json

INTAKE_PROMPT = Path(__file__).parents[1] / "prompts" / "science_intake.md"
DESIGN_PROMPT = Path(__file__).parents[1] / "prompts" / "experiment_design.md"

INTAKE_KEYS = ["conjecture_restated", "model", "signal_description",
               "control_description", "candidate_observables"]
DESIGN_KEYS = ["positive_experiment", "control_experiment",
               "criteria_positive", "criteria_control"]


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
    intake_prompt = INTAKE_PROMPT.read_text().format(
        catalog=catalog_for_prompt(), conjecture=conjecture)
    intake_proposal = ask_json(
        intake_prompt, required_keys=INTAKE_KEYS, agent="science-intake")

    model_name = intake_proposal["model"]
    if model_name not in CATALOG:
        raise ValueError(f"LLM proposed unknown model {model_name!r} "
                         f"(allowed: {list(CATALOG)})")
    spec = CATALOG[model_name]

    for obs in intake_proposal.get("candidate_observables", []):
        if obs not in OBSERVABLES:
            raise ValueError(f"intake proposed unknown observable {obs!r}")

    design_prompt = DESIGN_PROMPT.read_text().format(
        catalog=catalog_for_prompt(), intake=intake_proposal)
    proposal = ask_json(
        design_prompt, required_keys=DESIGN_KEYS, agent="experiment-designer")

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
        conjecture=str(intake_proposal["conjecture_restated"]),
        model=model_name,
        parameters={"source": "native_opencode_agents_v1"},
        criteria=criteria_positive)

    return hypothesis, criteria_control, experiments
