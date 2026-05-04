"""Submission interface for the Gobblecube ETA Challenge.

The grader imports this module and calls predict(request) row-by-row. All
state needed at inference lives in model.pkl; no network or data files are
required inside the container.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from features import build_single_features

MODEL_PATH = Path(__file__).parent / "model.pkl"

_BUNDLE = None


def _load_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        with open(MODEL_PATH, "rb") as f:
            _BUNDLE = pickle.load(f)
    return _BUNDLE


def _predict_with_bundle(request: dict, bundle: dict) -> float:
    artifacts = bundle["artifacts"]
    x = build_single_features(request, artifacts).reshape(1, -1)

    model_pred = float(bundle["model"].predict(x)[0])
    prior_idx = artifacts["feature_index"]["pair_hour_prior_duration"]
    prior_pred = float(x[0, prior_idx])
    blend = float(bundle.get("blend_weight", 1.0))
    pred = blend * model_pred + (1.0 - blend) * prior_pred

    same_idx = artifacts["feature_index"]["same_zone"]
    if x[0, same_idx] > 0.5 and bundle.get("same_zone_model") is not None:
        same_pred = float(bundle["same_zone_model"].predict(x)[0])
        same_blend = float(bundle.get("same_zone_blend_weight", 1.0))
        pred = same_blend * same_pred + (1.0 - same_blend) * prior_pred

    if not np.isfinite(pred):
        pred = float(artifacts["global_median_duration"])
    return float(max(30.0, min(pred, 3.0 * 3600.0)))


def predict(request: dict) -> float:
    """Predict trip duration in seconds."""
    return _predict_with_bundle(request, _load_bundle())
