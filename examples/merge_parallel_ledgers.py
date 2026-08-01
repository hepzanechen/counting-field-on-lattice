"""Merge ledger.json + auto_experiments session summaries from parallel worker
worktrees into this worktree's canonical data/virtual_lab/.

Safe by construction: run ONLY after all parallel demo_auto_experiments processes
have exited (no worker is mid-write), and only ever reads other workers' files and
writes this worktree's own -- never writes into another worktree's data/.

Hypothesis IDs are timestamp-based (AUTO-<stamp>-<i>), so cross-worker collisions are
effectively impossible in practice, but this script asserts on collision rather than
silently overwriting, since a silent merge bug here would be exactly the kind of data
loss parallelization was designed to avoid.

Run from the primary worktree:
  uv run python -m examples.merge_parallel_ledgers ../countingFieldOnLattice-infra-w2 ../countingFieldOnLattice-infra-w3 ../countingFieldOnLattice-infra-w4
"""
import argparse
import json
import shutil
from pathlib import Path

LEDGER_REL = Path("data/virtual_lab/ledger.json")
AUTO_REL = Path("data/virtual_lab/auto_experiments")


def load_records(ledger_path: Path) -> dict:
    if not ledger_path.exists():
        return {}
    data = json.loads(ledger_path.read_text())
    return {r["id"]: r for r in data.get("records", [])}


def main():
    parser = argparse.ArgumentParser(description="Merge parallel worker ledgers")
    parser.add_argument("worker_dirs", nargs="+", help="paths to other worker worktrees")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    primary_records = load_records(LEDGER_REL)
    before = len(primary_records)
    added = []

    for wd in args.worker_dirs:
        worker_path = Path(wd)
        worker_ledger = worker_path / LEDGER_REL
        worker_records = load_records(worker_ledger)
        for rid, rec in worker_records.items():
            if rid in primary_records:
                if primary_records[rid] != rec:
                    raise RuntimeError(
                        f"ID collision with DIFFERING content for {rid} "
                        f"(from {worker_path}) -- refusing to merge, needs manual review")
                continue  # identical record, already merged in a prior run
            primary_records[rid] = rec
            added.append((rid, str(worker_path)))

        # copy auto_experiments session summaries too (for benchmark_virtual_lab.py)
        worker_auto = worker_path / AUTO_REL
        if worker_auto.exists() and not args.dry_run:
            AUTO_REL.mkdir(parents=True, exist_ok=True)
            for session_dir in worker_auto.iterdir():
                dest = AUTO_REL / f"{worker_path.name}__{session_dir.name}"
                if not dest.exists():
                    shutil.copytree(session_dir, dest)

    print(f"Primary had {before} records; merging in {len(added)} new records:")
    for rid, src in added:
        print(f"  {rid}  <- {src}")

    if args.dry_run:
        print("(dry run, nothing written)")
        return

    LEDGER_REL.write_text(json.dumps({
        "records": list(primary_records.values())
    }, indent=2, default=str))
    print(f"Wrote merged ledger with {len(primary_records)} total records to {LEDGER_REL}")


if __name__ == "__main__":
    main()
