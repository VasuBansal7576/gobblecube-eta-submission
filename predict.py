"""Submission interface for the Gobblecube ETA Challenge.

The required public interface is predict(request). The Docker entrypoint uses
predict_many(...) for fast file-level scoring with the same feature path. All
state needed at inference lives in model.pkl; no network or data files are
required inside the container.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_single_features
from train import add_time_and_route, build_feature_frame

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
    return float(predict_many([request])[0])


def predict_many(requests: pd.DataFrame | list[dict]) -> np.ndarray:
    """Vectorized prediction path used by the Docker grader entrypoint."""
    bundle = _load_bundle()
    artifacts = bundle["artifacts"]
    df = pd.DataFrame(requests).copy()
    for col in ("pickup_zone", "dropoff_zone"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 265).astype(np.int16)

    meta = {
        "zone_is_manhattan": artifacts["zone_is_manhattan"],
        "zone_is_airport": artifacts["zone_is_airport"],
    }
    df = add_time_and_route(df, meta)
    x = build_feature_frame(df, artifacts)
    x_np = x.to_numpy(np.float32)

    model_pred = bundle["model"].predict(x_np)
    prior_pred = x["pair_hour_prior_duration"].to_numpy(np.float32)
    blend = float(bundle.get("blend_weight", 1.0))
    pred = blend * model_pred + (1.0 - blend) * prior_pred

    same_model = bundle.get("same_zone_model")
    same_mask = x["same_zone"].to_numpy() > 0.5
    if same_model is not None and np.any(same_mask):
        same_pred = same_model.predict(x_np[same_mask])
        same_blend = float(bundle.get("same_zone_blend_weight", 1.0))
        pred[same_mask] = same_blend * same_pred + (1.0 - same_blend) * prior_pred[same_mask]

    pred = np.nan_to_num(pred, nan=float(artifacts["global_median_duration"]))
    return np.clip(pred, 30.0, 3.0 * 3600.0).astype(np.float64)
