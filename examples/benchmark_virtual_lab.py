"""Aggregate performance snapshot of the Virtual Lab from data already on disk.

Reuses existing artifacts rather than re-running anything:
  - data/virtual_lab/ledger.json               (hypotheses, evidence, judge_reports)
  - data/virtual_lab/auto_experiments/*/session_summary.json  (per-session cycle timing)

Reports the metrics that matter for "is the science agent working, not just running":
  - cycle completion rate (real verdict vs. ERROR/timeout) and mean cycle time by outcome
  - per-invariant pass rate, overall and by model
  - hypothesis status distribution
  - which registered models have actually been exercised

Run:
  uv run python -m examples.benchmark_virtual_lab
  uv run python -m examples.benchmark_virtual_lab --append doc/virtual-lab-log/benchmark.md
"""
import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

LEDGER_PATH = Path("data/virtual_lab/ledger.json")
AUTO_DIR = Path("data/virtual_lab/auto_experiments")


def _load_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    return json.loads(LEDGER_PATH.read_text()).get("records", [])


def _load_sessions() -> list[dict]:
    sessions = []
    if not AUTO_DIR.exists():
        return sessions
    for summary_file in sorted(AUTO_DIR.glob("*/session_summary.json")):
        try:
            sessions.append(json.loads(summary_file.read_text()))
        except json.JSONDecodeError:
            continue
    return sessions


def cycle_stats(sessions: list[dict]) -> dict:
    all_results = [r for s in sessions for r in s.get("results", [])]
    total = len(all_results)
    errored = [r for r in all_results if str(r["decision"]).startswith("ERROR")]
    completed = [r for r in all_results if not str(r["decision"]).startswith("ERROR")]

    def _mean_time(rows):
        times = [r["cycle_time_s"] for r in rows if "cycle_time_s" in r]
        return round(statistics.mean(times), 1) if times else None

    error_kinds = Counter()
    for r in errored:
        d = str(r["decision"])
        if "literature-cartographer" in d:
            error_kinds["literature-cartographer timeout"] += 1
        elif "integrator" in d:
            error_kinds["integrator timeout"] += 1
        elif "creative-explorer" in d:
            error_kinds["creative-explorer timeout"] += 1
        elif "TimeoutExpired" in d:
            error_kinds["other timeout"] += 1
        else:
            error_kinds["other error"] += 1

    return {
        "total_cycles": total,
        "completed": len(completed),
        "errored": len(errored),
        "completion_rate": round(len(completed) / total, 3) if total else None,
        "mean_time_completed_s": _mean_time(completed),
        "mean_time_errored_s": _mean_time(errored),
        "error_breakdown": dict(error_kinds),
        "verdict_distribution": dict(Counter(str(r["decision"]) for r in completed)),
    }


def invariant_stats(records: list[dict]) -> dict:
    by_invariant = Counter()
    by_invariant_pass = Counter()
    by_model_invariant = defaultdict(Counter)
    by_model_invariant_pass = defaultdict(Counter)
    status_dist = Counter()
    model_dist = Counter()

    for rec in records:
        status_dist[rec.get("status", "?")] += 1
        model = rec.get("model", "?")
        model_dist[model] += 1
        for ev in rec.get("evidence", []):
            for inv in ev.get("judge_report", []):
                name = inv.get("invariant", "?")
                by_invariant[name] += 1
                by_model_invariant[model][name] += 1
                if inv.get("passed"):
                    by_invariant_pass[name] += 1
                    by_model_invariant_pass[model][name] += 1

    pass_rate = {
        name: round(by_invariant_pass[name] / count, 3)
        for name, count in by_invariant.items()
    }
    pass_rate_by_model = {
        model: {
            name: round(by_model_invariant_pass[model][name] / count, 3)
            for name, count in inv_counts.items()
        }
        for model, inv_counts in by_model_invariant.items()
    }

    return {
        "hypothesis_status_distribution": dict(status_dist),
        "model_distribution": dict(model_dist),
        "invariant_pass_rate": pass_rate,
        "invariant_pass_rate_by_model": pass_rate_by_model,
        "total_hypotheses": len(records),
    }


def main():
    parser = argparse.ArgumentParser(description="Virtual Lab performance snapshot")
    parser.add_argument("--append", type=str, default=None,
                        help="markdown file to append a dated summary row to")
    args = parser.parse_args()

    records = _load_ledger()
    sessions = _load_sessions()

    snapshot = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cycles": cycle_stats(sessions),
        "hypotheses": invariant_stats(records),
    }

    print(json.dumps(snapshot, indent=2))

    if args.append:
        c, h = snapshot["cycles"], snapshot["hypotheses"]
        row = (
            f"| {snapshot['timestamp']} "
            f"| {c['completed']}/{c['total_cycles']} ({c['completion_rate']}) "
            f"| {c['mean_time_completed_s']}s / {c['mean_time_errored_s']}s "
            f"| {c['error_breakdown']} "
            f"| {h['hypothesis_status_distribution']} "
            f"| {h['model_distribution']} |\n"
        )
        path = Path(args.append)
        if not path.exists():
            path.write_text(
                "# Virtual Lab benchmark log\n\n"
                "Auto-appended by `examples.benchmark_virtual_lab --append`. "
                "Each row is a cumulative snapshot as of that timestamp (not a delta).\n\n"
                "| timestamp | cycles completed/total (rate) | mean time completed/errored | "
                "error breakdown | hypothesis status dist | model dist |\n"
                "|---|---|---|---|---|---|\n"
            )
        with path.open("a") as f:
            f.write(row)
        print(f"\nAppended snapshot row to {path}")


if __name__ == "__main__":
    main()
