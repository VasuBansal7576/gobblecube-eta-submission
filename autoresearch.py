#!/usr/bin/env python
"""Small metric-driven experiment loop for the ETA submission.

This is the practical version of the AutoResearch idea used for this take-home:
run one named modeling change at a time, measure Dev MAE, append a ledger row,
and only promote an artifact when it beats the current best metric.

The script intentionally avoids any inference-time APIs. It shells out to
train.py with explicit flags so every experiment is reproducible from the git
log and research ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent
RESEARCH_DIR = ROOT / "research_runs"
LEDGER_PATH = ROOT / "research_log.csv"
BEST_MODEL_PATH = ROOT / "model.pkl"
BEST_METRICS_PATH = ROOT / "metrics.json"


@dataclass
class Experiment:
    name: str
    args: list[str] = field(default_factory=list)


DEFAULT_EXPERIMENTS = [
    Experiment("baseline_1m", ["--sample-n", "1000000", "--max-iter", "260"]),
    Experiment("no_density_1m", ["--sample-n", "1000000", "--max-iter", "260", "--disable-feature-group", "density"]),
    Experiment("no_ratecode_priors_1m", ["--sample-n", "1000000", "--max-iter", "260", "--disable-feature-group", "ratecode_priors"]),
    Experiment("no_neighbor_1m", ["--sample-n", "1000000", "--max-iter", "260", "--disable-feature-group", "neighbor"]),
    Experiment("no_recency_1m", ["--sample-n", "1000000", "--max-iter", "260", "--recency-half-life-days", "0"]),
    Experiment("squared_error_1m", ["--sample-n", "1000000", "--max-iter", "260", "--loss", "squared_error"]),
    Experiment(
        "squared_error_no_recency_1m",
        ["--sample-n", "1000000", "--max-iter", "260", "--loss", "squared_error", "--recency-half-life-days", "0"],
    ),
    Experiment(
        "squared_error_no_density_1m",
        ["--sample-n", "1000000", "--max-iter", "260", "--loss", "squared_error", "--disable-feature-group", "density"],
    ),
    Experiment(
        "squared_error_no_cap_1m",
        ["--sample-n", "1000000", "--max-iter", "260", "--loss", "squared_error", "--target-cap-quantile", "1.0"],
    ),
    Experiment(
        "squared_error_no_cap_1m_180",
        ["--sample-n", "1000000", "--max-iter", "180", "--loss", "squared_error", "--target-cap-quantile", "1.0"],
    ),
    Experiment(
        "squared_error_no_cap_1m_340",
        ["--sample-n", "1000000", "--max-iter", "340", "--loss", "squared_error", "--target-cap-quantile", "1.0"],
    ),
    Experiment(
        "squared_error_no_cap_2m_260",
        ["--sample-n", "2000000", "--max-iter", "260", "--loss", "squared_error", "--target-cap-quantile", "1.0"],
    ),
    Experiment("absolute_error_1m", ["--sample-n", "1000000", "--max-iter", "260", "--loss", "absolute_error"]),
    Experiment("no_same_zone_model_1m", ["--sample-n", "1000000", "--max-iter", "260", "--no-same-zone-model"]),
]


def load_best_score() -> float:
    if not BEST_METRICS_PATH.exists():
        return float("inf")
    with open(BEST_METRICS_PATH) as f:
        return float(json.load(f)["dev_mae"])


def append_ledger(row: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LEDGER_PATH.exists()
    with open(LEDGER_PATH, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "status",
                "dev_mae",
                "best_before",
                "promoted",
                "elapsed_seconds",
                "metrics_path",
                "args",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(exp: Experiment, promote: bool) -> dict:
    RESEARCH_DIR.mkdir(exist_ok=True)
    model_path = RESEARCH_DIR / f"{exp.name}.pkl"
    metrics_path = RESEARCH_DIR / f"{exp.name}.json"
    best_before = load_best_score()
    cmd = [
        sys.executable,
        "train.py",
        "--experiment-name",
        exp.name,
        "--model-path",
        str(model_path),
        "--metrics-path",
        str(metrics_path),
        *exp.args,
    ]

    print("\n==>", " ".join(cmd))
    t0 = time.time()
    status = "ok"
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        with open(metrics_path) as f:
            metrics = json.load(f)
        dev_mae = float(metrics["dev_mae"])
        promoted = bool(promote and dev_mae < best_before)
        if promoted:
            shutil.copy2(model_path, BEST_MODEL_PATH)
            shutil.copy2(metrics_path, BEST_METRICS_PATH)
            print(f"PROMOTED {exp.name}: {dev_mae:.3f} < {best_before:.3f}")
        else:
            print(f"kept current best: {dev_mae:.3f} vs {best_before:.3f}")
    except Exception:
        status = "failed"
        dev_mae = float("nan")
        promoted = False
        raise
    finally:
        elapsed = round(time.time() - t0, 1)
        append_ledger(
            {
                "experiment": exp.name,
                "status": status,
                "dev_mae": dev_mae,
                "best_before": best_before,
                "promoted": promoted,
                "elapsed_seconds": elapsed,
                "metrics_path": metrics_path,
                "args": " ".join(exp.args),
            }
        )
    return {"name": exp.name, "dev_mae": dev_mae, "promoted": promoted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true", help="copy a better model/metrics into the submission")
    parser.add_argument(
        "--only",
        action="append",
        help="run only experiments whose names match this value; may be repeated",
    )
    args = parser.parse_args()

    experiments = DEFAULT_EXPERIMENTS
    if args.only:
        names = set(args.only)
        experiments = [exp for exp in experiments if exp.name in names]
        missing = names - {exp.name for exp in experiments}
        if missing:
            raise SystemExit(f"Unknown experiments: {sorted(missing)}")

    results = [run_experiment(exp, promote=args.promote) for exp in experiments]
    print("\nSummary")
    for result in results:
        print(f"  {result['name']}: {result['dev_mae']:.3f} promoted={result['promoted']}")
    print(f"\nLedger: {LEDGER_PATH}")


if __name__ == "__main__":
    main()
