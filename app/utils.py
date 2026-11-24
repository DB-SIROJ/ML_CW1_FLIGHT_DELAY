"""Cached loaders and helpers shared by all Streamlit pages."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (ALL_FEATURES, CARRIER_NAMES, HIGH_CARD_CATEGORICAL,  # noqa: E402
                          LOW_CARD_CATEGORICAL, NUMERIC_FEATURES, TARGET, _time_block)

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM"]
COLORS = {"Logistic Regression": "#4C72B0", "Decision Tree": "#DD8452",
          "Random Forest": "#55A868", "LightGBM": "#C44E52", "Baseline (prior)": "#8C8C8C"}


@st.cache_data(show_spinner="Loading flight data…")
def load_sample() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "flights_sample.parquet")
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    return df


@st.cache_data
def load_airports() -> pd.DataFrame:
    return pd.read_csv(PROC / "airports_lookup.csv")


@st.cache_data
def load_routes() -> pd.DataFrame:
    return pd.read_csv(PROC / "routes.csv")


@st.cache_data
def load_metadata() -> dict:
    return json.loads((MODELS / "metadata.json").read_text())


@st.cache_data
def load_metric_tables():
    test = pd.read_csv(MODELS / "test_metrics.csv", index_col=0)
    val = pd.read_csv(MODELS / "validation_metrics.csv", index_col=0)
    ci = pd.read_csv(MODELS / "bootstrap_ci.csv", index_col=0)
    split = pd.read_csv(MODELS / "split_comparison.csv", index_col=0)
    imp = pd.read_csv(MODELS / "permutation_importance.csv", index_col=0).iloc[:, 0]
    preds = pd.read_parquet(MODELS / "test_predictions_sample.parquet")
    return test, val, ci, split, imp, preds


@st.cache_resource(show_spinner="Loading trained model…")
def load_model():
    return joblib.load(MODELS / "final_model.joblib")


@st.cache_data
def lookup_tables():
    """Typical schedule values used to fill features the user does not type in."""
    df = load_sample()
    return {
        "route_distance": load_routes().assign(ROUTE=lambda r: r.ORIGIN + "_" + r.DEST).set_index("ROUTE")["DISTANCE"],
        "route_duration": df.groupby("ROUTE")["SCHED_DURATION"].median(),
        "origin_hour": df.groupby(["ORIGIN", "DEP_HOUR"])["ORIGIN_HOURLY_DEPS"].median(),
        "origin_any": df.groupby("ORIGIN")["ORIGIN_HOURLY_DEPS"].median(),
        "dest_hour": df.groupby(["DEST", "ARR_HOUR"])["DEST_HOURLY_ARRS"].median(),
        "dest_any": df.groupby("DEST")["DEST_HOURLY_ARRS"].median(),
        "origin_daily": df.groupby("ORIGIN")["ORIGIN_DAILY_DEPS"].median(),
        "carrier_share": df.groupby(["ORIGIN", "OP_UNIQUE_CARRIER"])["CARRIER_DAILY_SHARE"].median(),
    }


@st.cache_data
def load_prev_rates() -> pd.DataFrame:
    """Previous-day disruption shares computed from the FULL dataset (all 1.6M flights).

    Keyed by the date they apply to: the row for 16 March holds the outcome of 15 March.
    """
    d = pd.read_parquet(PROC / "prev_day_rates.parquet")
    d["FL_DATE"] = pd.to_datetime(d["FL_DATE"])
    return d


PREV_DATE_MIN, PREV_DATE_MAX = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-04-01")


def prev_day_rates(origin: str, dest: str, carrier: str, date) -> tuple[dict, bool]:
    """Real previous-day rates for dates covered by the data; otherwise each entity's typical (median) value.

    Returns ({'origin','dest','carrier','national'}, is_actual).
    """
    d = load_prev_rates()
    date = pd.Timestamp(date)
    keys = {"origin": ("airport_origin", origin), "dest": ("airport_dest", dest),
            "carrier": ("carrier", carrier), "national": ("national", "ALL")}
    out, actual = {}, PREV_DATE_MIN <= date <= PREV_DATE_MAX
    for name, (lvl, key) in keys.items():
        s = d[(d.LEVEL == lvl) & (d.KEY == key)]
        typical = float(s.RATE.median()) if len(s) else float(d.loc[d.LEVEL == lvl, "RATE"].median())
        if actual:
            row = s[s.FL_DATE == date]
            out[name] = float(row.RATE.iloc[0]) if len(row) else typical   # airport had no flights that day
        else:
            out[name] = typical
    return out, actual


CONDITION_LEVELS = {  # user-facing description -> percentile of each entity's historical daily disruption
    "🌤️ Calm": 0.10,
    "⛅ Normal": 0.50,
    "🌧️ Disrupted": 0.85,
    "⛈️ Severe": 0.97,
}


def condition_rates(origin: str, dest: str, carrier: str, level: str) -> dict:
    """Translate a plain description of current conditions into disruption shares.

    Used for dates outside the data (e.g. tomorrow). Each entity gets the chosen percentile of its OWN
    history, so 'Disrupted' at a storm-prone airport means more than at a calm one.
    """
    d = load_prev_rates()
    q = CONDITION_LEVELS[level]
    keys = {"origin": ("airport_origin", origin), "dest": ("airport_dest", dest),
            "carrier": ("carrier", carrier), "national": ("national", "ALL")}
    out = {}
    for name, (lvl, key) in keys.items():
        s = d[(d.LEVEL == lvl) & (d.KEY == key) & (d.N_PREV_DAY >= 5)].RATE
        if len(s) < 10:
            s = d.loc[d.LEVEL == lvl, "RATE"]
        out[name] = float(s.quantile(q))
    return out


def airport_label(code: str) -> str:
    """'ATL — Hartsfield-Jackson Atlanta International Airport (GA)'."""
    ap = load_airports().set_index("IATA")
    if code in ap.index:
        r = ap.loc[code]
        return f"{code} — {r.AIRPORT_NAME} ({r.STATE})"
    return code


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def _get(series, key, default):
    try:
        v = series.loc[key]
        return float(v) if pd.notna(v) else default
    except KeyError:
        return default


def build_feature_row(carrier, origin, dest, date, dep_hhmm, arr_hhmm,
                      origin_prev=None, dest_prev=None, carrier_prev=None, national_prev=None,
                      origin_hourly=None):
    """Turn a user's flight description into the exact feature vector the model expects."""
    lk = lookup_tables()
    ap = load_airports().set_index("IATA")
    prev, _ = prev_day_rates(origin, dest, carrier, date)   # real values where the data covers the date
    o, d = ap.loc[origin], ap.loc[dest]
    route = f"{origin}_{dest}"
    dep_min = (dep_hhmm // 100) * 60 + dep_hhmm % 100
    arr_min = (arr_hhmm // 100) * 60 + arr_hhmm % 100
    dep_h, arr_h = dep_min // 60, arr_min // 60
    dur = arr_min - dep_min
    distance = _get(lk["route_distance"], route,
                    haversine_miles(o.LAT, o.LON, d.LAT, d.LON))
    dow = pd.Timestamp(date).dayofweek
    ohd = origin_hourly if origin_hourly is not None else _get(
        lk["origin_hour"], (origin, dep_h), _get(lk["origin_any"], origin, 3))
    dha = _get(lk["dest_hour"], (dest, arr_h), _get(lk["dest_any"], dest, 3))
    row = {
        "DEP_HOUR_SIN": np.sin(2 * np.pi * dep_h / 24), "DEP_HOUR_COS": np.cos(2 * np.pi * dep_h / 24),
        "ARR_HOUR_SIN": np.sin(2 * np.pi * arr_h / 24), "ARR_HOUR_COS": np.cos(2 * np.pi * arr_h / 24),
        "DAY_OF_WEEK": dow, "IS_WEEKEND": int(dow >= 5),
        "LOG_DISTANCE": np.log1p(distance),
        "SCHED_DURATION": dur + 1440 if dur < 0 else dur, "OVERNIGHT": int(dur < 0),
        "LOG_ORIGIN_HOURLY_DEPS": np.log1p(ohd), "LOG_DEST_HOURLY_ARRS": np.log1p(dha),
        "LOG_ORIGIN_DAILY_DEPS": np.log1p(_get(lk["origin_daily"], origin, 30)),
        "CARRIER_DAILY_SHARE": _get(lk["carrier_share"], (origin, carrier), 0.2),
        "ORIGIN_LAT": o.LAT, "ORIGIN_LON": o.LON, "ORIGIN_ELEV": o.ELEV,
        "DEST_LAT": d.LAT, "DEST_LON": d.LON, "DEST_ELEV": d.ELEV,
        "ORIGIN_IS_LARGE": int(o.IS_LARGE), "DEST_IS_LARGE": int(d.IS_LARGE),
        "ORIGIN_PREV_DAY_RATE": origin_prev if origin_prev is not None else prev["origin"],
        "DEST_PREV_DAY_RATE": dest_prev if dest_prev is not None else prev["dest"],
        "CARRIER_PREV_DAY_RATE": carrier_prev if carrier_prev is not None else prev["carrier"],
        "NATIONAL_PREV_DAY_RATE": national_prev if national_prev is not None else prev["national"],
        "OP_UNIQUE_CARRIER": carrier, "DEP_TIME_BLOCK": _time_block(pd.Series([dep_h])).iloc[0],
        "ORIGIN": origin, "DEST": dest, "ROUTE": route,
        "ORIGIN_STATE": o.STATE, "DEST_STATE": d.STATE,
    }
    return pd.DataFrame([row])[ALL_FEATURES], distance


def page_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)
