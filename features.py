"""Shared ETA feature logic for training and inference.

The grader only supplies pickup/dropoff zone, requested_at, and
passenger_count. Every feature here is either derived from those fields or from
offline artifacts built only from the 2023 training data.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

AIRPORT_ZONES = {1, 132, 138}

# Approximate Manhattan CBD / dense core taxi zones. This is intentionally
# structural, not a 2024-specific event list.
CBD_ZONES = {
    4, 12, 13, 24, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114,
    125, 137, 140, 141, 142, 143, 144, 148, 158, 161, 162, 163, 164,
    170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 246, 249,
    261, 262, 263,
}

ROUTE_SAME_ZONE = 0
ROUTE_AIRPORT = 1
ROUTE_MANHATTAN_INTERNAL = 2
ROUTE_MANHATTAN_OUTER = 3
ROUTE_OUTER_OUTER = 4
N_ROUTE_CLASSES = 5

FEATURE_NAMES = [
    "pickup_zone",
    "dropoff_zone",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "qhour_sin",
    "qhour_cos",
    "is_weekend",
    "is_rush_hour",
    "is_late_night",
    "is_federal_holiday",
    "is_holiday_eve",
    "is_december_holiday_period",
    "is_nye_period",
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
    "pair_prior_duration",
    "pair_hour_prior_duration",
    "route_hour_duration",
    "cluster_hour_duration",
    "pickup_hour_duration",
    "dropoff_hour_duration",
    "neighbor_pickup_hour_duration",
    "neighbor_dropoff_hour_duration",
    "distance_prior_miles",
    "speed_prior_mph",
    "physics_duration",
    "pickup_density_15m",
    "dropoff_density_15m",
    "log_pair_count",
    "log_pair_hour_count",
    "prob_rate_jfk",
    "prob_rate_newark",
    "prob_rate_negotiated",
]


def zone_to_int(value: object) -> int:
    try:
        z = int(value)
    except Exception:
        return 0
    if 1 <= z <= 265:
        return z
    return 0


def route_class(
    pickup_zone: int,
    dropoff_zone: int,
    zone_is_manhattan: np.ndarray,
    zone_is_airport: np.ndarray,
) -> int:
    if pickup_zone == dropoff_zone:
        return ROUTE_SAME_ZONE
    if bool(zone_is_airport[pickup_zone]) or bool(zone_is_airport[dropoff_zone]):
        return ROUTE_AIRPORT
    p_m = bool(zone_is_manhattan[pickup_zone])
    d_m = bool(zone_is_manhattan[dropoff_zone])
    if p_m and d_m:
        return ROUTE_MANHATTAN_INTERNAL
    if p_m or d_m:
        return ROUTE_MANHATTAN_OUTER
    return ROUTE_OUTER_OUTER


def route_class_array(
    pickup_zone: np.ndarray,
    dropoff_zone: np.ndarray,
    zone_is_manhattan: np.ndarray,
    zone_is_airport: np.ndarray,
) -> np.ndarray:
    pz = pickup_zone.astype(np.int16, copy=False)
    dz = dropoff_zone.astype(np.int16, copy=False)
    out = np.full(len(pz), ROUTE_OUTER_OUTER, dtype=np.int8)
    same = pz == dz
    airport = zone_is_airport[pz] | zone_is_airport[dz]
    p_m = zone_is_manhattan[pz]
    d_m = zone_is_manhattan[dz]
    out[p_m | d_m] = ROUTE_MANHATTAN_OUTER
    out[p_m & d_m] = ROUTE_MANHATTAN_INTERNAL
    out[airport] = ROUTE_AIRPORT
    out[same] = ROUTE_SAME_ZONE
    return out


def cyc(value: float, period: float) -> tuple[float, float]:
    angle = 2.0 * np.pi * float(value) / float(period)
    return float(np.sin(angle)), float(np.cos(angle))


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(days=7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        d = date(year, 12, 31)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def federal_holidays(year: int) -> set[date]:
    fixed = [
        date(year, 1, 1),
        date(year, 6, 19),
        date(year, 7, 4),
        date(year, 11, 11),
        date(year, 12, 25),
    ]
    variable = [
        _nth_weekday(year, 1, 0, 3),   # MLK Day
        _nth_weekday(year, 2, 0, 3),   # Presidents' Day
        _last_weekday(year, 5, 0),     # Memorial Day
        _nth_weekday(year, 9, 0, 1),   # Labor Day
        _nth_weekday(year, 10, 0, 2),  # Columbus / Indigenous Peoples' Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
    ]
    holidays = set(variable)
    for d in fixed:
        holidays.add(d)
        holidays.add(_observed(d))
    return holidays


def calendar_flags(ts: datetime) -> tuple[float, float, float, float]:
    d = ts.date()
    holidays = federal_holidays(ts.year)
    # Include observed New Year's Day from adjacent years around Jan 1.
    holidays |= federal_holidays(ts.year - 1)
    holidays |= federal_holidays(ts.year + 1)
    is_holiday = d in holidays
    is_eve = d + timedelta(days=1) in holidays
    is_december_period = (ts.month == 12 and ts.day >= 15) or (ts.month == 1 and ts.day <= 2)
    is_nye_period = (ts.month == 12 and ts.day == 31) or (ts.month == 1 and ts.day == 1)
    return (
        float(is_holiday),
        float(is_eve),
        float(is_december_period),
        float(is_nye_period),
    )


def _safe(value: float, fallback: float) -> float:
    if np.isfinite(value) and value > 0:
        return float(value)
    return float(fallback)


def _shrink(child: float, count: float, parent: float, k: float) -> float:
    if not np.isfinite(child) or child <= 0 or count <= 0:
        return float(parent)
    return float((count * child + k * parent) / (count + k))


def build_single_features(request: dict, artifacts: dict) -> np.ndarray:
    pz = zone_to_int(request.get("pickup_zone"))
    dz = zone_to_int(request.get("dropoff_zone"))
    ts = datetime.fromisoformat(str(request["requested_at"]))

    hour = ts.hour
    dow = ts.weekday()
    doy = ts.timetuple().tm_yday
    qhour = hour * 4 + ts.minute // 15

    zone_is_airport = artifacts["zone_is_airport"]
    zone_is_cbd = artifacts["zone_is_cbd"]
    zone_is_manhattan = artifacts["zone_is_manhattan"]
    rclass = route_class(pz, dz, zone_is_manhattan, zone_is_airport)

    global_median = float(artifacts["global_median_duration"])
    route_hour = _safe(artifacts["route_hour_duration"][rclass, hour], global_median)

    cluster = int(artifacts["pair_cluster"][pz, dz])
    if cluster >= 0:
        cluster_hour = _safe(artifacts["cluster_hour_duration"][cluster, hour], route_hour)
    else:
        cluster_hour = route_hour

    pair_count = float(artifacts["pair_count"][pz, dz])
    pair_hour_count = float(artifacts["pair_hour_count"][pz, dz, hour])
    pair_prior = _shrink(
        float(artifacts["pair_median_duration"][pz, dz]),
        pair_count,
        cluster_hour,
        float(artifacts["pair_shrink_k"]),
    )
    pair_hour_prior = _shrink(
        float(artifacts["pair_hour_median_duration"][pz, dz, hour]),
        pair_hour_count,
        pair_prior,
        float(artifacts["pair_hour_shrink_k"]),
    )

    route_dist = _safe(artifacts["route_class_distance"][rclass], artifacts["global_distance_miles"])
    dist_prior = _shrink(
        float(artifacts["pair_median_distance"][pz, dz]),
        pair_count,
        route_dist,
        float(artifacts["pair_shrink_k"]),
    )

    route_speed = _safe(artifacts["route_hour_speed"][rclass, hour], artifacts["global_speed_mph"])
    speed_prior = _shrink(
        float(artifacts["pair_median_speed"][pz, dz]),
        pair_count,
        route_speed,
        float(artifacts["pair_shrink_k"]),
    )
    speed_prior = max(speed_prior, 1.0)
    physics_duration = float(dist_prior / speed_prior * 3600.0)

    pickup_hour = _safe(artifacts["pickup_hour_duration"][pz, hour], route_hour)
    dropoff_hour = _safe(artifacts["dropoff_hour_duration"][dz, hour], route_hour)

    p_neighbors = artifacts["zone_neighbors"][pz]
    d_neighbors = artifacts["zone_neighbors"][dz]
    neighbor_pickup = _safe(
        float(np.nanmean(artifacts["pickup_hour_duration"][p_neighbors, hour])),
        pickup_hour,
    )
    neighbor_dropoff = _safe(
        float(np.nanmean(artifacts["dropoff_hour_duration"][d_neighbors, hour])),
        dropoff_hour,
    )

    p_density = float(artifacts["pickup_density"][pz, dow, qhour])
    d_density = float(artifacts["dropoff_density"][dz, dow, qhour])
    rate_probs = artifacts["pair_rate_probs"][pz, dz]

    h_sin, h_cos = cyc(hour + ts.minute / 60.0, 24.0)
    dow_sin, dow_cos = cyc(dow, 7.0)
    doy_sin, doy_cos = cyc(doy, 366.0 if ts.year % 4 == 0 else 365.0)
    q_sin, q_cos = cyc(qhour, 96.0)
    is_holiday, is_eve, is_dec, is_nye = calendar_flags(ts)

    values = [
        float(pz),
        float(dz),
        h_sin,
        h_cos,
        dow_sin,
        dow_cos,
        doy_sin,
        doy_cos,
        q_sin,
        q_cos,
        float(dow >= 5),
        float((7 <= hour <= 10) or (16 <= hour <= 19)),
        float(hour <= 5 or hour >= 22),
        is_holiday,
        is_eve,
        is_dec,
        is_nye,
        float(pz == dz),
        float(bool(zone_is_airport[pz]) or bool(zone_is_airport[dz])),
        float(zone_is_airport[pz]),
        float(zone_is_airport[dz]),
        float(zone_is_cbd[pz]),
        float(zone_is_cbd[dz]),
        float(bool(zone_is_cbd[pz]) or bool(zone_is_cbd[dz])),
        float(zone_is_manhattan[pz]),
        float(zone_is_manhattan[dz]),
        float(rclass),
        pair_prior,
        pair_hour_prior,
        route_hour,
        cluster_hour,
        pickup_hour,
        dropoff_hour,
        neighbor_pickup,
        neighbor_dropoff,
        dist_prior,
        speed_prior,
        physics_duration,
        np.log1p(p_density),
        np.log1p(d_density),
        np.log1p(pair_count),
        np.log1p(pair_hour_count),
        float(rate_probs[0]),
        float(rate_probs[1]),
        float(rate_probs[2]),
    ]
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
