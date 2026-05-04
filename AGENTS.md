# Agent Notes

This submission was built with Codex as the coding agent. The agent was asked
to optimize for a working, reproducible Gobblecube ETA submission rather than a
prototype.

Core constraints to preserve:

- `predict.py` must expose `predict(request: dict) -> float`.
- The grader only supplies `pickup_zone`, `dropoff_zone`, `requested_at`, and
  `passenger_count`.
- No external network calls are allowed at inference time.
- `trip_distance` and `ratecode_id` may be used only to build offline training
  priors; they must not be required by `predict()`.
- The scoring metric is MAE, so median/quantile-oriented training is preferred
  over squared-error defaults when it improves Dev MAE.

Implementation notes:

- `train.py` creates all offline artifacts and writes `model.pkl`.
- `features.py` contains shared inference-safe feature construction.
- `predict.py` lazily loads `model.pkl` and does no file/network access beyond
  that local pickle.
- `metrics.json` records the latest Dev score, ablations, segmented MAE, and
  residual diagnostics after training.
