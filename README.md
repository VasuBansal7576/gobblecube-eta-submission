# Gobblecube ETA Challenge Submission

This repo builds a Dockerized ETA predictor for the Gobblecube ride-hailing
take-home. The grader calls:

```python
predict({
    "pickup_zone": int,
    "dropoff_zone": int,
    "requested_at": "YYYY-MM-DDTHH:MM:SS",
    "passenger_count": int,
}) -> float  # duration_seconds
```

No network or external data files are used at inference time. Everything needed
by `predict.py` is stored in `model.pkl`.

## Final Dev Score

- `python grade.py` 50k Dev sample MAE: **253.3 seconds**
- Full local Dev MAE from `train.py`: **250.8 seconds**
- Docker image size: **966MB**

Starter reference from the challenge README: naive GBT baseline is about
`351s` Dev / `367s` Eval, and a simple zone-pair lookup is about `300s` Dev.

## Approach

The core model treats ETA as a structured tabular problem rather than a raw
zone-id regression. I build full-year 2023 priors for route duration, observed
distance, speed, demand density, fare-regime likelihood, and route class. I
started with quantile loss because Gobblecube scores MAE, but the ablation loop
found that a squared-error `HistGradientBoostingRegressor` on the cleaned
feature stack scored better on Dev. The shipped model uses the measured winner,
not the initial theory.

Main pieces:

- **Measured objective choice:** quantile, absolute-error, and squared-error
  losses were all tested; squared-error won on Dev for the final feature stack.
- **Empirical Bayes route priors:** zone-pair and zone-pair-hour medians shrink
  toward speed-regime cluster means, not the global average.
- **Observed distance/speed:** `trip_distance` and `RatecodeID` are used only
  during training to build compact priors; `predict()` never requires them.
- **Cyclical time features:** hour, day-of-week, day-of-year, and 15-minute
  bucket encoded with sin/cos, plus weekend/rush/late-night flags.
- **Calendar structure:** federal holidays, holiday eves, December holiday
  period, and NYE/New Year period.
- **NYC route structure:** airport flags, Manhattan/CBD flags, route-class
  flags, and a separate same-zone model.
- **Demand proxy:** historical pickup/dropoff trip density by zone, day, and
  15-minute bucket.
- **Geographic borrowing:** nearest-zone aggregate features from the TLC taxi
  zone shapefile.
- **Recency weighting:** late-2023 rows receive more weight than early-2023
  rows, while older rows still help rare routes.
- **Target cleaning tested, not assumed:** p99.5 route-class winsorization was
  implemented and tested; the final model keeps the raw cleaned target because
  the no-cap experiment won Dev MAE.
- **Passenger count skepticism:** passenger count is retained for cleaning but
  deliberately excluded from model features because it is driver-entered and
  default-heavy.

The final prediction uses the squared-error model directly, except same-zone
trips where the dedicated same-zone model is lightly blended with the
zone-pair-hour prior. Blend weights are selected on Dev.

## Ablations

Measured on full local Dev inside `train.py`:

| Model / prior | MAE |
|---|---:|
| Pair median prior | 299.9s |
| Distance / speed physics prior | 300.4s |
| Pair-hour shrinkage prior | 273.8s |
| Initial 3M-row quantile blend | 253.8s |
| 1M-row quantile control | 252.1s |
| 1M-row squared-error | 251.5s |
| 1M-row squared-error, no target cap | 251.2s |
| Final 1M-row squared-error, no target cap, 340 iters | **250.8s** |

The metric-driven loop is in `autoresearch.py`, with results in
`research_log.csv` and per-run JSON files in `research_runs/`. A larger 2M-row
variant and a 6M-row earlier run both scored worse than the final 1M-row model,
so the smaller model is intentional.

## Diagnostics

Segmented MAE from the selected model:

| Segment | MAE |
|---|---:|
| Overall | 250.8s |
| Same-zone | 209.1s |
| Manhattan internal | 216.3s |
| Airport route | 428.5s |
| Manhattan to/from outer borough | 405.9s |
| Outer-to-outer | 537.4s |
| Rush hour | 264.8s |
| Late night | 194.9s |

Residual analysis shows remaining error is concentrated in afternoon peak
hours, airport routes, outer-borough routes, and dropoffs into zone `265`
("Unknown"). The next useful work would target those segments rather than
blindly adding more global features.

## What Did Not Work

- Training on a larger 6M-row weighted sample worsened Dev MAE, likely because
  the tree started fitting older/noisier regimes instead of the cleaner
  recency-weighted signal.
- The metric argument for quantile/absolute-error loss was directionally
  sensible but empirically wrong here. Squared-error scored better once the
  target, priors, and features were in place.
- p99.5 target winsorization also lost to training on the raw cleaned target.
- A pure distance/speed physics prior was not enough by itself. Observed
  distance is useful as a feature, but route duration still needs historical
  priors and a flexible model.
- Treating all same-zone trips like normal route pairs produced unstable
  distance/speed behavior, so same-zone examples use a dedicated model path.

## AI Tooling

I used Codex as an implementation partner for repo reading, feature-pipeline
construction, verification, and the ablation loop. The useful loop was:
propose one modeling change, encode it in `train.py`, run `autoresearch.py`,
compare Dev MAE, and promote only if the metric improved. The loop overturned
two plausible assumptions: quantile loss and winsorization both sounded right,
but squared-error with no target cap scored better.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Downloads public 2023 TLC yellow taxi parquet files and builds train/dev.
python data/download_data.py

# Reproduce the final promoted model.pkl and metrics.json.
python train.py \
  --experiment-name squared_error_no_cap_1m_340 \
  --sample-n 1000000 \
  --max-iter 340 \
  --loss squared_error \
  --target-cap-quantile 1.0

# Optional: rerun the recorded ablation loop.
python autoresearch.py --promote

# Local contract and scoring checks.
python -m pytest tests/
python grade.py

# Docker grader pathway.
docker build -t gobblecube-eta-submission .
docker run --rm --network=none -v $(pwd)/data:/work \
  gobblecube-eta-submission /work/dev.parquet /work/preds.csv
```

## External Data

Only public NYC TLC data is used:

- 2023 NYC Yellow Taxi trip records, downloaded by `data/download_data.py`
- NYC TLC taxi zone lookup CSV
- NYC TLC taxi zone shapefile for nearest-zone geographic borrowing

No external API calls are made at inference time.

Total active build time: about one focused work session.
