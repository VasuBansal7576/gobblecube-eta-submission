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
MAX_TRAIN_SECONDS = 360.0
LATE_TIME_TOLERANCE_SECONDS = 1.0
SEGMENT_TOLERANCE_SECONDS = 2.0
LEDGER_FIELDS = [
    "experiment",
    "status",
    "dev_mae",
    "best_before",
    "late_time_mae",
    "best_late_before",
    "promoted",
    "gate_reason",
    "elapsed_seconds",
    "metrics_path",
    "args",
]


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
        "squared_error_no_cap_1m_420",
        ["--sample-n", "1000000", "--max-iter", "420", "--loss", "squared_error", "--target-cap-quantile", "1.0"],
    ),
    Experiment(
        "squared_error_no_cap_1m_520",
        ["--sample-n", "1000000", "--max-iter", "520", "--loss", "squared_error", "--target-cap-quantile", "1.0"],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_340",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "340",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl30_1m_340",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "340",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "30",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl60_1m_340",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "340",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "60",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_420",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "420",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_520",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "520",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_620",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "620",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_700",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "700",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_800",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "800",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl38_1m_620",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "620",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "38",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl36_1m_620",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "620",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "36",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl40_1m_620",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "620",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "40",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl38_1m_620_leaf31",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "620",
            "--max-leaf-nodes",
            "31",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "38",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl38_1m_700_leaf31",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "700",
            "--max-leaf-nodes",
            "31",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "38",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl37_1m_700_leaf31",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "700",
            "--max-leaf-nodes",
            "31",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "37",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl39_1m_700_leaf31",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "700",
            "--max-leaf-nodes",
            "31",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "39",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl38_1m_760_leaf31",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "760",
            "--max-leaf-nodes",
            "31",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "38",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl38_1m_820_leaf31",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "820",
            "--max-leaf-nodes",
            "31",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "38",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl38_1m_700_leaf47",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "700",
            "--max-leaf-nodes",
            "47",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "38",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl38_1m_700_leaf31_min120",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "700",
            "--max-leaf-nodes",
            "31",
            "--min-samples-leaf",
            "120",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "38",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl52_1m_620",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "620",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "52",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_700_lr045",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "700",
            "--learning-rate",
            "0.045",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_800_lr040",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "800",
            "--learning-rate",
            "0.040",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_620_leaf95",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "620",
            "--max-leaf-nodes",
            "95",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_620_leaf31",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "620",
            "--max-leaf-nodes",
            "31",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_620_min40",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "620",
            "--min-samples-leaf",
            "40",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl45_1m_620_min140",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "620",
            "--min-samples-leaf",
            "140",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "45",
        ],
    ),
    Experiment(
        "squared_error_no_cap_hl150_1m_340",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "340",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--recency-half-life-days",
            "150",
        ],
    ),
    Experiment(
        "squared_error_no_cap_no_rate_1m_340",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "340",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--disable-feature-group",
            "ratecode_priors",
        ],
    ),
    Experiment(
        "squared_error_no_cap_no_neighbor_1m_340",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "340",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--disable-feature-group",
            "neighbor",
        ],
    ),
    Experiment(
        "squared_error_no_cap_no_samezone_1m_340",
        [
            "--sample-n",
            "1000000",
            "--max-iter",
            "340",
            "--loss",
            "squared_error",
            "--target-cap-quantile",
            "1.0",
            "--no-same-zone-model",
        ],
    ),
    Experiment(
        "squared_error_no_cap_2m_260",
        ["--sample-n", "2000000", "--max-iter", "260", "--loss", "squared_error", "--target-cap-quantile", "1.0"],
    ),
    Experiment("absolute_error_1m", ["--sample-n", "1000000", "--max-iter", "260", "--loss", "absolute_error"]),
    Experiment("no_same_zone_model_1m", ["--sample-n", "1000000", "--max-iter", "260", "--no-same-zone-model"]),
]


def load_best_metrics() -> dict:
    if not BEST_METRICS_PATH.exists():
        return {"dev_mae": float("inf")}
    with open(BEST_METRICS_PATH) as f:
        return json.load(f)


def metric_value(metrics: dict, name: str) -> float:
    value = metrics.get(name)
    if value is None:
        return float("inf")
    return float(value)


def promotion_decision(metrics: dict, best_metrics: dict, promote: bool) -> tuple[bool, str]:
    if not promote:
        return False, "promotion disabled"
    dev_mae = metric_value(metrics, "dev_mae")
    best_dev = metric_value(best_metrics, "dev_mae")
    if dev_mae >= best_dev:
        return False, f"dev {dev_mae:.3f} >= best {best_dev:.3f}"

    elapsed = metric_value(metrics, "elapsed_seconds")
    if elapsed > MAX_TRAIN_SECONDS:
        return False, f"train runtime {elapsed:.1f}s > {MAX_TRAIN_SECONDS:.1f}s"

    late_mae = metrics.get("late_time_mae")
    best_late = best_metrics.get("late_time_mae")
    if late_mae is not None and best_late is not None:
        if float(late_mae) > float(best_late) + LATE_TIME_TOLERANCE_SECONDS:
            return False, f"late-time {float(late_mae):.3f} regressed vs {float(best_late):.3f}"

    segments = metrics.get("segment_mae", {})
    best_segments = best_metrics.get("segment_mae", {})
    for name, value in segments.items():
        if name == "overall" or name not in best_segments:
            continue
        if float(value) > float(best_segments[name]) + SEGMENT_TOLERANCE_SECONDS:
            return False, f"segment {name} regressed {float(value):.3f} vs {float(best_segments[name]):.3f}"

    return True, "passed gate"


def normalize_ledger() -> None:
    if not LEDGER_PATH.exists():
        return
    with open(LEDGER_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames == LEDGER_FIELDS:
            return
    with open(LEDGER_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LEDGER_FIELDS})


def append_ledger(row: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalize_ledger()
    exists = LEDGER_PATH.exists()
    with open(LEDGER_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(exp: Experiment, promote: bool) -> dict:
    RESEARCH_DIR.mkdir(exist_ok=True)
    model_path = RESEARCH_DIR / f"{exp.name}.pkl"
    metrics_path = RESEARCH_DIR / f"{exp.name}.json"
    best_metrics = load_best_metrics()
    best_before = metric_value(best_metrics, "dev_mae")
    best_late_before = best_metrics.get("late_time_mae", "")
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
        late_time_mae = metrics.get("late_time_mae", "")
        promoted, gate_reason = promotion_decision(metrics, best_metrics, promote)
        if promoted:
            shutil.copy2(model_path, BEST_MODEL_PATH)
            shutil.copy2(metrics_path, BEST_METRICS_PATH)
            print(f"PROMOTED {exp.name}: {dev_mae:.3f} < {best_before:.3f}")
        else:
            print(f"kept current best: {dev_mae:.3f} vs {best_before:.3f} ({gate_reason})")
    except Exception:
        status = "failed"
        dev_mae = float("nan")
        late_time_mae = ""
        promoted = False
        gate_reason = "failed"
        raise
    finally:
        elapsed = round(time.time() - t0, 1)
        append_ledger(
            {
                "experiment": exp.name,
                "status": status,
                "dev_mae": dev_mae,
                "best_before": best_before,
                "late_time_mae": late_time_mae,
                "best_late_before": best_late_before,
                "promoted": promoted,
                "gate_reason": gate_reason,
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
