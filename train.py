#!/usr/bin/env python
"""Train the ETA submission model.

This is intentionally a practical tabular pipeline, not a black box:

* label-side priors are built only from 2023 training data
* loss function, target capping, and recency choices are ablated by metric
* recent rows get larger sample weights to reduce 2023 -> 2024 drift
* same-zone trips are modeled separately because route distance collapses
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor

from features import (
    AIRPORT_ZONES,
    CBD_ZONES,
    FEATURE_NAMES,
    N_ROUTE_CLASSES,
    calendar_flags,
    cyc,
    route_class_array,
)

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_PATH = Path(__file__).parent / "model.pkl"
METRICS_PATH = Path(__file__).parent / "metrics.json"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONE_SHAPE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

PAIR_SHRINK_K = 80.0
PAIR_HOUR_SHRINK_K = 35.0
N_CLUSTERS = 8

FEATURE_GROUPS = {
    "calendar": [
        "is_federal_holiday",
        "is_holiday_eve",
        "is_december_holiday_period",
        "is_nye_period",
    ],
    "density": ["pickup_density_15m", "dropoff_density_15m"],
    "ratecode_priors": ["prob_rate_jfk", "prob_rate_newark", "prob_rate_negotiated"],
    "neighbor": ["neighbor_pickup_hour_duration", "neighbor_dropoff_hour_duration"],
    "physics": ["distance_prior_miles", "speed_prior_mph", "physics_duration"],
    "cyclical_time": [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "qhour_sin",
        "qhour_cos",
    ],
    "structure": [
        "same_zone",
        "airport_route",
        "pickup_airport",
        "dropoff_airport",
        "pickup_cbd",
        "dropoff_cbd",
        "touches_cbd",
        "pickup_manhattan",
        "dropoff_manhattan",
        "route_class",
    ],
    "cluster": ["cluster_hour_duration"],
}


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"  fetching {url}")
        urlretrieve(url, path)
    return path


def load_zone_metadata() -> dict:
    lookup_path = _download(ZONE_LOOKUP_URL, RAW_DIR / "taxi_zone_lookup.csv")
    lookup = pd.read_csv(lookup_path)

    borough_to_id = {b: i for i, b in enumerate(sorted(lookup["Borough"].dropna().unique()))}
    zone_borough_id = np.zeros(266, dtype=np.int8)
    zone_is_manhattan = np.zeros(266, dtype=bool)
    zone_is_airport = np.zeros(266, dtype=bool)
    zone_is_cbd = np.zeros(266, dtype=bool)

    for row in lookup.itertuples(index=False):
        loc = int(row.LocationID)
        if 1 <= loc <= 265:
            borough = str(row.Borough)
            zone_borough_id[loc] = borough_to_id.get(borough, 0)
            zone_is_manhattan[loc] = borough == "Manhattan"
            zone_is_airport[loc] = loc in AIRPORT_ZONES
            zone_is_cbd[loc] = loc in CBD_ZONES

    return {
        "borough_to_id": borough_to_id,
        "zone_borough_id": zone_borough_id,
        "zone_is_manhattan": zone_is_manhattan,
        "zone_is_airport": zone_is_airport,
        "zone_is_cbd": zone_is_cbd,
        "zone_neighbors": build_zone_neighbors(),
    }


def build_zone_neighbors(k: int = 5) -> np.ndarray:
    """Nearest-zone fallback for geographic borrowing.

    The plan called this adjacency; nearest centroids are a lightweight proxy
    that keeps inference simple while still letting rare zones borrow spatial
    traffic signal from nearby zones.
    """
    neighbors = np.tile(np.arange(266, dtype=np.int16).reshape(-1, 1), (1, k))
    try:
        import shapefile  # type: ignore
    except Exception:
        return neighbors

    zip_path = _download(ZONE_SHAPE_URL, RAW_DIR / "taxi_zones.zip")
    extract_dir = RAW_DIR / "taxi_zones"
    shp_path = extract_dir / "taxi_zones.shp"
    if not shp_path.exists() and not list(extract_dir.glob("**/taxi_zones.shp")):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    if not shp_path.exists():
        matches = list(extract_dir.glob("**/taxi_zones.shp"))
        if matches:
            shp_path = matches[0]

    centroids = np.full((266, 2), np.nan, dtype=np.float64)
    reader = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in reader.fields[1:]]
    loc_idx = fields.index("LocationID")
    for record in reader.shapeRecords():
        loc = int(record.record[loc_idx])
        if not (1 <= loc <= 265):
            continue
        pts = np.asarray(record.shape.points, dtype=np.float64)
        if len(pts):
            centroids[loc] = pts.mean(axis=0)

    valid = np.where(np.isfinite(centroids[:, 0]))[0]
    for loc in range(1, 266):
        if not np.isfinite(centroids[loc, 0]):
            continue
        dist = np.hypot(centroids[valid, 0] - centroids[loc, 0], centroids[valid, 1] - centroids[loc, 1])
        order = valid[np.argsort(dist)[:k]]
        neighbors[loc, : len(order)] = order
    return neighbors


def add_time_and_route(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["requested_at"])
    out["_ts"] = ts
    out["hour"] = ts.dt.hour.astype("int8")
    out["dow"] = ts.dt.dayofweek.astype("int8")
    out["doy"] = ts.dt.dayofyear.astype("int16")
    out["qhour"] = (ts.dt.hour * 4 + ts.dt.minute // 15).astype("int8")
    pz = out["pickup_zone"].to_numpy(dtype=np.int16)
    dz = out["dropoff_zone"].to_numpy(dtype=np.int16)
    out["route_class"] = route_class_array(
        pz,
        dz,
        meta["zone_is_manhattan"],
        meta["zone_is_airport"],
    )
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), cutoff)])


def add_recency_weights(df: pd.DataFrame, half_life_days: float = 90.0, floor: float = 0.30) -> np.ndarray:
    max_ts = df["_ts"].max()
    age_days = (max_ts - df["_ts"]).dt.days.to_numpy(dtype=np.float32)
    if half_life_days <= 0:
        return np.ones(len(df), dtype=np.float32)
    return (floor + (1.0 - floor) * np.exp(-age_days / half_life_days)).astype(np.float32)


def valid_speed_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trip_distance" not in out.columns:
        out["trip_distance"] = np.nan
    dist = out["trip_distance"].astype("float32")
    dur = out["duration_seconds"].astype("float32")
    speed = dist / np.maximum(dur / 3600.0, 1e-6)
    out["speed_mph"] = speed.where((dist >= 0.05) & (dist <= 150) & (speed >= 1) & (speed <= 90))
    out["valid_distance"] = dist.where((dist >= 0.05) & (dist <= 150))
    return out


def route_class_caps(train: pd.DataFrame, quantile: float) -> np.ndarray:
    caps = np.full(N_ROUTE_CLASSES, np.nan, dtype=np.float32)
    if quantile >= 1.0:
        caps.fill(float(train["duration_seconds"].max()))
        return caps
    q = train.groupby("route_class")["duration_seconds"].quantile(quantile)
    global_cap = float(train["duration_seconds"].quantile(quantile))
    for cls in range(N_ROUTE_CLASSES):
        caps[cls] = float(q.get(cls, global_cap))
    return caps


def fill_2d_stat(grouped: pd.DataFrame, col: str, dtype=np.float32) -> np.ndarray:
    arr = np.full((266, 266), np.nan, dtype=dtype)
    for row in grouped[["pickup_zone", "dropoff_zone", col]].itertuples(index=False):
        arr[int(row.pickup_zone), int(row.dropoff_zone)] = getattr(row, col)
    return arr


def fill_2d_count(grouped: pd.DataFrame, col: str) -> np.ndarray:
    arr = np.zeros((266, 266), dtype=np.uint32)
    for row in grouped[["pickup_zone", "dropoff_zone", col]].itertuples(index=False):
        arr[int(row.pickup_zone), int(row.dropoff_zone)] = int(getattr(row, col))
    return arr


def fill_3d_stat(grouped: pd.DataFrame, col: str, dtype=np.float32) -> np.ndarray:
    arr = np.full((266, 266, 24), np.nan, dtype=dtype)
    for row in grouped[["pickup_zone", "dropoff_zone", "hour", col]].itertuples(index=False):
        arr[int(row.pickup_zone), int(row.dropoff_zone), int(row.hour)] = getattr(row, col)
    return arr


def fill_3d_count(grouped: pd.DataFrame, col: str) -> np.ndarray:
    arr = np.zeros((266, 266, 24), dtype=np.uint16)
    clipped = grouped[["pickup_zone", "dropoff_zone", "hour", col]].copy()
    clipped[col] = clipped[col].clip(0, np.iinfo(np.uint16).max)
    for row in clipped.itertuples(index=False):
        arr[int(row.pickup_zone), int(row.dropoff_zone), int(row.hour)] = int(getattr(row, col))
    return arr


def build_artifacts(train: pd.DataFrame, meta: dict) -> dict:
    print("Building priors and route structure...")
    train = valid_speed_frame(train)
    global_median = float(train["duration_seconds"].median())
    global_distance = float(train["valid_distance"].median())
    global_speed = float(train["speed_mph"].median())

    pair = train.groupby(["pickup_zone", "dropoff_zone"], observed=True).agg(
        pair_count=("duration_seconds", "size"),
        pair_median_duration=("duration_seconds", "median"),
        pair_median_distance=("valid_distance", "median"),
        pair_median_speed=("speed_mph", "median"),
        prob_rate_jfk=("ratecode_id", lambda s: float((s == 2).mean())),
        prob_rate_newark=("ratecode_id", lambda s: float((s == 3).mean())),
        prob_rate_negotiated=("ratecode_id", lambda s: float((s == 5).mean())),
    ).reset_index()
    pair_hour = train.groupby(["pickup_zone", "dropoff_zone", "hour"], observed=True).agg(
        pair_hour_count=("duration_seconds", "size"),
        pair_hour_median_duration=("duration_seconds", "median"),
        pair_hour_median_speed=("speed_mph", "median"),
    ).reset_index()

    pair_count = fill_2d_count(pair, "pair_count")
    pair_median_duration = fill_2d_stat(pair, "pair_median_duration")
    pair_median_distance = fill_2d_stat(pair, "pair_median_distance")
    pair_median_speed = fill_2d_stat(pair, "pair_median_speed")
    pair_hour_count = fill_3d_count(pair_hour, "pair_hour_count")
    pair_hour_median_duration = fill_3d_stat(pair_hour, "pair_hour_median_duration")
    pair_hour_median_speed = fill_3d_stat(pair_hour, "pair_hour_median_speed")

    pair_rate_probs = np.zeros((266, 266, 3), dtype=np.float32)
    for row in pair.itertuples(index=False):
        pair_rate_probs[int(row.pickup_zone), int(row.dropoff_zone)] = [
            float(row.prob_rate_jfk),
            float(row.prob_rate_newark),
            float(row.prob_rate_negotiated),
        ]

    route_hour_duration = np.full((N_ROUTE_CLASSES, 24), global_median, dtype=np.float32)
    route_hour_speed = np.full((N_ROUTE_CLASSES, 24), global_speed, dtype=np.float32)
    route_hour = train.groupby(["route_class", "hour"], observed=True).agg(
        duration=("duration_seconds", "median"),
        speed=("speed_mph", "median"),
    ).reset_index()
    for row in route_hour.itertuples(index=False):
        route_hour_duration[int(row.route_class), int(row.hour)] = float(row.duration)
        if np.isfinite(row.speed):
            route_hour_speed[int(row.route_class), int(row.hour)] = float(row.speed)

    route_class_distance = np.full(N_ROUTE_CLASSES, global_distance, dtype=np.float32)
    route_dist = train.groupby("route_class", observed=True)["valid_distance"].median()
    for cls in range(N_ROUTE_CLASSES):
        val = route_dist.get(cls, global_distance)
        route_class_distance[cls] = float(val if np.isfinite(val) else global_distance)

    pickup_hour_duration = np.full((266, 24), global_median, dtype=np.float32)
    dropoff_hour_duration = np.full((266, 24), global_median, dtype=np.float32)
    for row in train.groupby(["pickup_zone", "hour"], observed=True)["duration_seconds"].median().reset_index().itertuples(index=False):
        pickup_hour_duration[int(row.pickup_zone), int(row.hour)] = float(row.duration_seconds)
    for row in train.groupby(["dropoff_zone", "hour"], observed=True)["duration_seconds"].median().reset_index().itertuples(index=False):
        dropoff_hour_duration[int(row.dropoff_zone), int(row.hour)] = float(row.duration_seconds)

    pickup_density = np.zeros((266, 7, 96), dtype=np.float32)
    dropoff_density = np.zeros((266, 7, 96), dtype=np.float32)
    for row in train.groupby(["pickup_zone", "dow", "qhour"], observed=True).size().reset_index(name="n").itertuples(index=False):
        pickup_density[int(row.pickup_zone), int(row.dow), int(row.qhour)] = float(row.n)
    for row in train.groupby(["dropoff_zone", "dow", "qhour"], observed=True).size().reset_index(name="n").itertuples(index=False):
        dropoff_density[int(row.dropoff_zone), int(row.dow), int(row.qhour)] = float(row.n)

    pair_cluster, cluster_hour_duration = build_speed_clusters(
        pair_count,
        pair_hour_median_speed,
        train,
        route_hour_duration,
    )

    artifacts = {
        "feature_names": FEATURE_NAMES,
        "feature_index": {name: i for i, name in enumerate(FEATURE_NAMES)},
        "global_median_duration": global_median,
        "global_distance_miles": global_distance,
        "global_speed_mph": global_speed,
        "pair_shrink_k": PAIR_SHRINK_K,
        "pair_hour_shrink_k": PAIR_HOUR_SHRINK_K,
        "pair_count": pair_count,
        "pair_median_duration": pair_median_duration,
        "pair_median_distance": pair_median_distance,
        "pair_median_speed": pair_median_speed,
        "pair_hour_count": pair_hour_count,
        "pair_hour_median_duration": pair_hour_median_duration,
        "pair_hour_median_speed": pair_hour_median_speed,
        "pair_rate_probs": pair_rate_probs,
        "route_hour_duration": route_hour_duration,
        "route_hour_speed": route_hour_speed,
        "route_class_distance": route_class_distance,
        "pickup_hour_duration": pickup_hour_duration,
        "dropoff_hour_duration": dropoff_hour_duration,
        "pickup_density": pickup_density,
        "dropoff_density": dropoff_density,
        "pair_cluster": pair_cluster,
        "cluster_hour_duration": cluster_hour_duration,
        "zone_neighbors": meta["zone_neighbors"],
        "zone_borough_id": meta["zone_borough_id"],
        "zone_is_manhattan": meta["zone_is_manhattan"],
        "zone_is_airport": meta["zone_is_airport"],
        "zone_is_cbd": meta["zone_is_cbd"],
    }
    return artifacts


def build_speed_clusters(
    pair_count: np.ndarray,
    pair_hour_median_speed: np.ndarray,
    train: pd.DataFrame,
    route_hour_duration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pair_cluster = np.full((266, 266), -1, dtype=np.int8)
    cluster_hour_duration = np.full((N_CLUSTERS, 24), np.nan, dtype=np.float32)
    pairs = np.argwhere(pair_count >= 80)
    if len(pairs) < N_CLUSTERS:
        return pair_cluster, cluster_hour_duration

    profiles = []
    valid_pairs = []
    global_hour_speed = np.nanmedian(pair_hour_median_speed, axis=(0, 1))
    global_hour_speed = np.nan_to_num(global_hour_speed, nan=float(np.nanmean(global_hour_speed)))
    for pz, dz in pairs:
        prof = pair_hour_median_speed[pz, dz].astype(np.float32)
        known = np.isfinite(prof)
        if known.sum() < 6:
            continue
        prof = np.where(known, prof, global_hour_speed)
        profiles.append(prof)
        valid_pairs.append((int(pz), int(dz)))
    if len(profiles) < N_CLUSTERS:
        return pair_cluster, cluster_hour_duration

    X = np.vstack(profiles)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    for (pz, dz), label in zip(valid_pairs, labels):
        pair_cluster[pz, dz] = int(label)

    row_clusters = pair_cluster[
        train["pickup_zone"].to_numpy(dtype=np.int16),
        train["dropoff_zone"].to_numpy(dtype=np.int16),
    ]
    tmp = train.loc[row_clusters >= 0, ["duration_seconds", "hour"]].copy()
    tmp["cluster"] = row_clusters[row_clusters >= 0]
    grouped = tmp.groupby(["cluster", "hour"], observed=True)["duration_seconds"].median().reset_index()
    for row in grouped.itertuples(index=False):
        cluster_hour_duration[int(row.cluster), int(row.hour)] = float(row.duration_seconds)
    for cluster in range(N_CLUSTERS):
        for hour in range(24):
            if not np.isfinite(cluster_hour_duration[cluster, hour]):
                cluster_hour_duration[cluster, hour] = float(np.nanmedian(route_hour_duration[:, hour]))
    return pair_cluster, cluster_hour_duration


def shrink(child: np.ndarray, count: np.ndarray, parent: np.ndarray, k: float) -> np.ndarray:
    child = child.astype(np.float32, copy=False)
    count = count.astype(np.float32, copy=False)
    parent = parent.astype(np.float32, copy=False)
    valid = np.isfinite(child) & (child > 0) & (count > 0)
    return np.where(valid, (count * child + k * parent) / (count + k), parent)


def build_feature_frame(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    pz = df["pickup_zone"].to_numpy(dtype=np.int16)
    dz = df["dropoff_zone"].to_numpy(dtype=np.int16)
    hour = df["hour"].to_numpy(dtype=np.int8)
    dow = df["dow"].to_numpy(dtype=np.int8)
    doy = df["doy"].to_numpy(dtype=np.int16)
    qhour = df["qhour"].to_numpy(dtype=np.int8)
    rclass = df["route_class"].to_numpy(dtype=np.int8)

    global_median = float(artifacts["global_median_duration"])
    pair_count = artifacts["pair_count"][pz, dz].astype(np.float32)
    pair_hour_count = artifacts["pair_hour_count"][pz, dz, hour].astype(np.float32)
    route_hour = artifacts["route_hour_duration"][rclass, hour].astype(np.float32)
    route_hour = np.nan_to_num(route_hour, nan=global_median)

    cluster = artifacts["pair_cluster"][pz, dz]
    cluster_hour = route_hour.copy()
    mask_cluster = cluster >= 0
    cluster_hour[mask_cluster] = artifacts["cluster_hour_duration"][cluster[mask_cluster], hour[mask_cluster]]
    cluster_hour = np.nan_to_num(cluster_hour, nan=global_median)

    pair_prior = shrink(
        artifacts["pair_median_duration"][pz, dz],
        pair_count,
        cluster_hour,
        float(artifacts["pair_shrink_k"]),
    )
    pair_hour_prior = shrink(
        artifacts["pair_hour_median_duration"][pz, dz, hour],
        pair_hour_count,
        pair_prior,
        float(artifacts["pair_hour_shrink_k"]),
    )

    route_dist = artifacts["route_class_distance"][rclass]
    dist_prior = shrink(
        artifacts["pair_median_distance"][pz, dz],
        pair_count,
        route_dist,
        float(artifacts["pair_shrink_k"]),
    )
    route_speed = artifacts["route_hour_speed"][rclass, hour]
    speed_prior = shrink(
        artifacts["pair_median_speed"][pz, dz],
        pair_count,
        route_speed,
        float(artifacts["pair_shrink_k"]),
    )
    speed_prior = np.maximum(np.nan_to_num(speed_prior, nan=float(artifacts["global_speed_mph"])), 1.0)
    physics_duration = dist_prior / speed_prior * 3600.0

    pickup_hour = artifacts["pickup_hour_duration"][pz, hour]
    dropoff_hour = artifacts["dropoff_hour_duration"][dz, hour]
    neigh = artifacts["zone_neighbors"]
    neighbor_pickup = np.nanmean(artifacts["pickup_hour_duration"][neigh[pz], hour[:, None]], axis=1)
    neighbor_dropoff = np.nanmean(artifacts["dropoff_hour_duration"][neigh[dz], hour[:, None]], axis=1)
    pickup_density = artifacts["pickup_density"][pz, dow, qhour]
    dropoff_density = artifacts["dropoff_density"][dz, dow, qhour]
    rate_probs = artifacts["pair_rate_probs"][pz, dz]

    ts = df["_ts"]
    hour_float = ts.dt.hour.to_numpy(dtype=np.float32) + ts.dt.minute.to_numpy(dtype=np.float32) / 60.0
    hour_sin = np.sin(2.0 * np.pi * hour_float / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour_float / 24.0)
    dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
    dow_cos = np.cos(2.0 * np.pi * dow / 7.0)
    doy_sin = np.sin(2.0 * np.pi * doy / 365.0)
    doy_cos = np.cos(2.0 * np.pi * doy / 365.0)
    q_sin = np.sin(2.0 * np.pi * qhour / 96.0)
    q_cos = np.cos(2.0 * np.pi * qhour / 96.0)

    flags = np.asarray([calendar_flags(x.to_pydatetime()) for x in ts], dtype=np.float32)
    zone_is_airport = artifacts["zone_is_airport"]
    zone_is_cbd = artifacts["zone_is_cbd"]
    zone_is_manhattan = artifacts["zone_is_manhattan"]

    values = {
        "pickup_zone": pz.astype(np.float32),
        "dropoff_zone": dz.astype(np.float32),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "qhour_sin": q_sin,
        "qhour_cos": q_cos,
        "is_weekend": (dow >= 5).astype(np.float32),
        "is_rush_hour": (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 19))).astype(np.float32),
        "is_late_night": ((hour <= 5) | (hour >= 22)).astype(np.float32),
        "is_federal_holiday": flags[:, 0],
        "is_holiday_eve": flags[:, 1],
        "is_december_holiday_period": flags[:, 2],
        "is_nye_period": flags[:, 3],
        "same_zone": (pz == dz).astype(np.float32),
        "airport_route": (zone_is_airport[pz] | zone_is_airport[dz]).astype(np.float32),
        "pickup_airport": zone_is_airport[pz].astype(np.float32),
        "dropoff_airport": zone_is_airport[dz].astype(np.float32),
        "pickup_cbd": zone_is_cbd[pz].astype(np.float32),
        "dropoff_cbd": zone_is_cbd[dz].astype(np.float32),
        "touches_cbd": (zone_is_cbd[pz] | zone_is_cbd[dz]).astype(np.float32),
        "pickup_manhattan": zone_is_manhattan[pz].astype(np.float32),
        "dropoff_manhattan": zone_is_manhattan[dz].astype(np.float32),
        "route_class": rclass.astype(np.float32),
        "pair_prior_duration": pair_prior,
        "pair_hour_prior_duration": pair_hour_prior,
        "route_hour_duration": route_hour,
        "cluster_hour_duration": cluster_hour,
        "pickup_hour_duration": pickup_hour,
        "dropoff_hour_duration": dropoff_hour,
        "neighbor_pickup_hour_duration": neighbor_pickup,
        "neighbor_dropoff_hour_duration": neighbor_dropoff,
        "distance_prior_miles": dist_prior,
        "speed_prior_mph": speed_prior,
        "physics_duration": physics_duration,
        "pickup_density_15m": np.log1p(pickup_density),
        "dropoff_density_15m": np.log1p(dropoff_density),
        "log_pair_count": np.log1p(pair_count),
        "log_pair_hour_count": np.log1p(pair_hour_count),
        "prob_rate_jfk": rate_probs[:, 0],
        "prob_rate_newark": rate_probs[:, 1],
        "prob_rate_negotiated": rate_probs[:, 2],
    }
    X = pd.DataFrame({name: values[name] for name in FEATURE_NAMES})
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    max_iter: int,
    loss: str,
) -> HistGradientBoostingRegressor:
    kwargs = {
        "loss": loss,
        "max_iter": max_iter,
        "learning_rate": 0.055,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 80,
        "l2_regularization": 0.02,
        "validation_fraction": None,
        "random_state": 42,
    }
    if loss == "quantile":
        kwargs["quantile"] = 0.5
    model = HistGradientBoostingRegressor(
        **kwargs,
    )
    model.fit(X.to_numpy(np.float32), y, sample_weight=weights)
    return model


def choose_training_rows(weights: np.ndarray, sample_n: int, seed: int = 42) -> np.ndarray:
    n = len(weights)
    if sample_n <= 0 or sample_n >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    probs = weights.astype(np.float64)
    probs = probs / probs.sum()
    return rng.choice(n, size=sample_n, replace=False, p=probs)


def disable_feature_groups(X: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    if not groups:
        return X
    out = X.copy()
    for group in groups:
        names = FEATURE_GROUPS.get(group)
        if names is None:
            raise SystemExit(f"Unknown feature group {group!r}. Known: {sorted(FEATURE_GROUPS)}")
        for name in names:
            if name in out.columns:
                out[name] = 0.0
    return out


def mae(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y_true)))


def tune_blend(y: np.ndarray, model_pred: np.ndarray, prior_pred: np.ndarray) -> tuple[float, float]:
    best_w = 1.0
    best = mae(y, model_pred)
    for w in np.linspace(0.0, 1.0, 21):
        pred = w * model_pred + (1.0 - w) * prior_pred
        score = mae(y, pred)
        if score < best:
            best = score
            best_w = float(w)
    return best_w, best


def segment_table(dev: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    masks = {
        "overall": np.ones(len(dev), dtype=bool),
        "same_zone": dev["pickup_zone"].to_numpy() == dev["dropoff_zone"].to_numpy(),
        "airport_route": dev["route_class"].to_numpy() == 1,
        "manhattan_internal": dev["route_class"].to_numpy() == 2,
        "manhattan_to_outer": dev["route_class"].to_numpy() == 3,
        "outer_to_outer": dev["route_class"].to_numpy() == 4,
        "rush_hour": dev["hour"].between(7, 10).to_numpy() | dev["hour"].between(16, 19).to_numpy(),
        "late_night": (dev["hour"] <= 5).to_numpy() | (dev["hour"] >= 22).to_numpy(),
    }
    out = {}
    for name, mask in masks.items():
        if mask.sum() > 0:
            out[name] = mae(y[mask], pred[mask])
    return out


def residual_summary(dev: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> dict:
    residual = np.abs(pred - y)
    summary = {}
    for col in ["hour", "dow", "route_class", "pickup_zone", "dropoff_zone"]:
        tmp = pd.DataFrame({col: dev[col].to_numpy(), "residual": residual})
        top = (
            tmp.groupby(col)["residual"]
            .agg(["mean", "count"])
            .query("count >= 100")
            .sort_values("mean", ascending=False)
            .head(8)
            .reset_index()
        )
        summary[col] = top.to_dict("records")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-n", type=int, default=3_000_000)
    parser.add_argument("--max-iter", type=int, default=420)
    parser.add_argument("--loss", choices=["quantile", "squared_error", "absolute_error"], default="quantile")
    parser.add_argument("--target-cap-quantile", type=float, default=0.995)
    parser.add_argument("--recency-half-life-days", type=float, default=90.0)
    parser.add_argument("--recency-floor", type=float, default=0.30)
    parser.add_argument("--disable-feature-group", action="append", default=[])
    parser.add_argument("--no-same-zone-model", action="store_true")
    parser.add_argument("--experiment-name", default="final")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--no-save-model", action="store_true")
    args = parser.parse_args()

    for path in (DATA_DIR / "train.parquet", DATA_DIR / "dev.parquet"):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run `python data/download_data.py` first.")

    t0 = time.time()
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")
    print(f"  train {len(train):,} rows; dev {len(dev):,} rows")

    meta = load_zone_metadata()
    train = add_time_and_route(train, meta)
    dev = add_time_and_route(dev, meta)

    print("Applying target winsorization and recency weights...")
    caps = route_class_caps(train, args.target_cap_quantile)
    y_train_raw = train["duration_seconds"].to_numpy(np.float32)
    y_train = np.minimum(y_train_raw, caps[train["route_class"].to_numpy(np.int8)])
    weights = add_recency_weights(train, args.recency_half_life_days, args.recency_floor)

    artifacts = build_artifacts(train, meta)
    artifacts["route_class_caps"] = caps

    rows = choose_training_rows(weights, args.sample_n)
    print("Featurizing sampled train rows and full dev...")
    X_train = build_feature_frame(train.iloc[rows].reset_index(drop=True), artifacts)
    X_dev = build_feature_frame(dev, artifacts)
    X_train = disable_feature_groups(X_train, args.disable_feature_group)
    X_dev = disable_feature_groups(X_dev, args.disable_feature_group)
    y_dev = dev["duration_seconds"].to_numpy(np.float32)

    print(f"Training {args.loss} model on {len(rows):,} rows...")
    model = train_model(X_train, y_train[rows], weights[rows], args.max_iter, args.loss)

    same_model = None
    same_mask_train = (train["pickup_zone"].to_numpy() == train["dropoff_zone"].to_numpy())
    if not args.no_same_zone_model and same_mask_train.sum() >= 5000:
        same_rows_all = np.where(same_mask_train)[0]
        same_rows = same_rows_all[:]
        if len(same_rows) > 600_000:
            same_rows = choose_training_rows(weights[same_rows], 600_000)
            same_rows = same_rows_all[same_rows]
        print(f"Training same-zone model on {len(same_rows):,} rows...")
        X_same = build_feature_frame(train.iloc[same_rows].reset_index(drop=True), artifacts)
        X_same = disable_feature_groups(X_same, args.disable_feature_group)
        same_model = train_model(X_same, y_train[same_rows], weights[same_rows], 260, args.loss)

    X_dev_np = X_dev.to_numpy(np.float32)
    model_pred = model.predict(X_dev_np)
    prior = X_dev["pair_hour_prior_duration"].to_numpy(np.float32)
    blend_w, blend_mae = tune_blend(y_dev, model_pred, prior)
    final_pred = blend_w * model_pred + (1.0 - blend_w) * prior

    same_blend_w = 1.0
    same_mask_dev = (dev["pickup_zone"].to_numpy() == dev["dropoff_zone"].to_numpy())
    if same_model is not None and same_mask_dev.sum() > 0:
        same_pred = same_model.predict(X_dev_np[same_mask_dev])
        same_prior = prior[same_mask_dev]
        same_blend_w, _ = tune_blend(y_dev[same_mask_dev], same_pred, same_prior)
        final_pred[same_mask_dev] = same_blend_w * same_pred + (1.0 - same_blend_w) * same_prior

    metrics = {
        "experiment_name": args.experiment_name,
        "dev_mae": mae(y_dev, final_pred),
        "model_only_mae": mae(y_dev, model_pred),
        "pair_hour_prior_mae": mae(y_dev, prior),
        "pair_prior_mae": mae(y_dev, X_dev["pair_prior_duration"].to_numpy(np.float32)),
        "physics_mae": mae(y_dev, X_dev["physics_duration"].to_numpy(np.float32)),
        "blend_weight": blend_w,
        "same_zone_blend_weight": same_blend_w,
        "segment_mae": segment_table(dev, y_dev, final_pred),
        "residual_summary": residual_summary(dev, y_dev, final_pred),
        "train_rows": int(len(train)),
        "model_train_rows": int(len(rows)),
        "loss": args.loss,
        "target_cap_quantile": args.target_cap_quantile,
        "recency_half_life_days": args.recency_half_life_days,
        "recency_floor": args.recency_floor,
        "disabled_feature_groups": args.disable_feature_group,
        "same_zone_model_enabled": same_model is not None,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(metrics, indent=2))

    bundle = {
        "model": model,
        "same_zone_model": same_model,
        "blend_weight": blend_w,
        "same_zone_blend_weight": same_blend_w,
        "artifacts": artifacts,
        "metrics": metrics,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {args.metrics_path}")
    if not args.no_save_model:
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.model_path, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {args.model_path}")


if __name__ == "__main__":
    main()
