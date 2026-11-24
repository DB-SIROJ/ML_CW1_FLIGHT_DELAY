"""Shared data-loading, cleaning and feature-engineering code.

Used by both the Jupyter notebook and the Streamlit app so that the two
always apply exactly the same transformations (single source of truth).
"""
from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXT_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MONTHS = ["January", "February", "March"]
TARGET = "ARR_DEL15"

# Lookup table for carrier codes (source: BTS "L_UNIQUE_CARRIERS" lookup).
CARRIER_NAMES = {
    "AA": "American Airlines", "AS": "Alaska Airlines", "B6": "JetBlue Airways",
    "DL": "Delta Air Lines", "F9": "Frontier Airlines", "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines", "MQ": "Envoy Air", "NK": "Spirit Airlines",
    "OH": "PSA Airlines", "OO": "SkyWest Airlines", "UA": "United Air Lines",
    "WN": "Southwest Airlines", "YX": "Republic Airline",
}

# Some IATA codes in the BTS data no longer match OurAirports because the
# airport has since been re-coded. Map BTS code -> OurAirports `ident`.
IATA_FIXES = {"PBI": "KPBI"}

# Feature groups used by the modelling pipeline.
# DAY_OF_MONTH and DAYS_TO_HOLIDAY are engineered for EDA but deliberately NOT
# used by the models (see notebook section 3.6: they memorise specific days and
# DAYS_TO_HOLIDAY is constant in the March validation/test period).
LAG_FEATURES = ["ORIGIN_PREV_DAY_RATE", "DEST_PREV_DAY_RATE",
                "CARRIER_PREV_DAY_RATE", "NATIONAL_PREV_DAY_RATE"]
NUMERIC_FEATURES = [
    "DEP_HOUR_SIN", "DEP_HOUR_COS", "ARR_HOUR_SIN", "ARR_HOUR_COS",
    "DAY_OF_WEEK", "IS_WEEKEND",
    "LOG_DISTANCE", "SCHED_DURATION", "OVERNIGHT",
    "LOG_ORIGIN_HOURLY_DEPS", "LOG_DEST_HOURLY_ARRS", "LOG_ORIGIN_DAILY_DEPS",
    "CARRIER_DAILY_SHARE",
    "ORIGIN_LAT", "ORIGIN_LON", "ORIGIN_ELEV", "DEST_LAT", "DEST_LON", "DEST_ELEV",
    "ORIGIN_IS_LARGE", "DEST_IS_LARGE",
] + LAG_FEATURES
LOW_CARD_CATEGORICAL = ["OP_UNIQUE_CARRIER", "DEP_TIME_BLOCK"]
HIGH_CARD_CATEGORICAL = ["ORIGIN", "DEST", "ROUTE", "ORIGIN_STATE", "DEST_STATE"]
ALL_FEATURES = NUMERIC_FEATURES + LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_flights(raw_dir: Path = RAW_DIR, months=MONTHS) -> pd.DataFrame:
    """Load and vertically concatenate the monthly BTS on-time files."""
    frames = []
    for m in months:
        path = raw_dir / f"Flights_2025_{m}.csv"
        if not path.exists():                      # repository stores gzip-compressed copies
            path = path.with_suffix(".csv.gz")
        f = pd.read_csv(path, low_memory=False,
                        dtype={"CRS_DEP_TIME": "int32", "CRS_ARR_TIME": "int32"})
        f["SOURCE_FILE"] = m
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def load_airports(path: Path = EXT_DIR / "airports.csv", codes=None) -> pd.DataFrame:
    """Load OurAirports metadata and return one row per BTS airport code."""
    a = pd.read_csv(path, low_memory=False,
                    usecols=["ident", "type", "name", "latitude_deg", "longitude_deg",
                             "elevation_ft", "iso_country", "iso_region", "iata_code"])
    by_iata = a.dropna(subset=["iata_code"]).drop_duplicates("iata_code").set_index("iata_code")
    if codes is None:
        codes = by_iata.index
    rows = []
    for c in codes:
        if c in IATA_FIXES:  # correct re-coded airports via their ICAO ident
            r = a.loc[a["ident"] == IATA_FIXES[c]].iloc[0].copy()
        elif c in by_iata.index:
            r = by_iata.loc[c].copy()
        else:
            continue
        r["IATA"] = c
        rows.append(r)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out = out.rename(columns={"latitude_deg": "LAT", "longitude_deg": "LON",
                              "elevation_ft": "ELEV", "name": "AIRPORT_NAME"})
    out["STATE"] = out["iso_region"].str.split("-").str[-1]
    out["IS_LARGE"] = (out["type"] == "large_airport").astype(int)
    return out[["IATA", "AIRPORT_NAME", "LAT", "LON", "ELEV", "STATE", "IS_LARGE"]]


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def hhmm_to_minutes(x: pd.Series) -> pd.Series:
    """Convert an HHMM integer (e.g. 1435) to minutes after midnight; 2400 -> 0."""
    x = x.where(x != 2400, 0)
    return (x // 100) * 60 + (x % 100)


def clean_flights(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fix data-quality problems. Returns the cleaned frame and a log of actions."""
    log = {"rows_in": len(df)}
    df = df.copy()

    # 1. exact duplicate records
    dup = df.duplicated(subset=[c for c in df.columns if c != "SOURCE_FILE"])
    log["exact_duplicates_removed"] = int(dup.sum())
    df = df.loc[~dup]

    # 2. parse the date (stored as text "1/1/2025 12:00:00 AM")
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")

    # 3. invalid HHMM clock values (minutes > 59 or hour > 24)
    bad_time = ((df["CRS_DEP_TIME"] % 100 > 59) | (df["CRS_DEP_TIME"] > 2400) |
                (df["CRS_ARR_TIME"] % 100 > 59) | (df["CRS_ARR_TIME"] > 2400))
    log["invalid_clock_times_removed"] = int(bad_time.sum())
    df = df.loc[~bad_time]

    # 4. impossible combinations
    impossible = (df["ORIGIN"] == df["DEST"]) | (df["DISTANCE"] <= 0)
    impossible |= (df["CANCELLED"] == 1) & (df["DIVERTED"] == 1)
    log["impossible_records_removed"] = int(impossible.sum())
    df = df.loc[~impossible]

    # 5. target consistency: ARR_DEL15 must equal 1{ARR_DELAY >= 15}
    has = df["ARR_DELAY"].notna()
    recomputed = (df.loc[has, "ARR_DELAY"] >= 15).astype(float)
    mism = recomputed != df.loc[has, "ARR_DEL15"]
    log["target_inconsistencies_corrected"] = int(mism.sum())
    df.loc[has, "ARR_DEL15"] = recomputed

    # 6. missing arrival delay must be explained by cancellation/diversion
    unexplained = df["ARR_DELAY"].isna() & (df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)
    log["unexplained_missing_target_removed"] = int(unexplained.sum())
    df = df.loc[~unexplained]

    log["rows_out"] = len(df)
    return df.reset_index(drop=True), log


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _time_block(hour: pd.Series) -> pd.Series:
    bins = [-1, 5, 11, 16, 20, 24]
    labels = ["Night(00-05)", "Morning(06-11)", "Afternoon(12-16)", "Evening(17-20)", "Late(21-23)"]
    return pd.cut(hour, bins=bins, labels=labels).astype(str)


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar, clock and congestion features. All are known before departure.

    Must be applied to the FULL schedule (including later-cancelled flights)
    because congestion is defined by what was scheduled, not what flew.
    """
    df = df.copy()
    dep_min = hhmm_to_minutes(df["CRS_DEP_TIME"])
    arr_min = hhmm_to_minutes(df["CRS_ARR_TIME"])
    df["DEP_HOUR"] = (dep_min // 60).astype(int)
    df["ARR_HOUR"] = (arr_min // 60).astype(int)
    for name, h in [("DEP", df["DEP_HOUR"]), ("ARR", df["ARR_HOUR"])]:
        df[f"{name}_HOUR_SIN"] = np.sin(2 * np.pi * h / 24)
        df[f"{name}_HOUR_COS"] = np.cos(2 * np.pi * h / 24)
    df["DEP_TIME_BLOCK"] = _time_block(df["DEP_HOUR"])

    df["MONTH"] = df["FL_DATE"].dt.month
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek          # 0 = Monday
    df["DAY_OF_MONTH"] = df["FL_DATE"].dt.day
    df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 5).astype(int)

    hol = np.array(sorted(pd.to_datetime(list(holidays.US(years=[2024, 2025]).keys()))),
                   dtype="datetime64[ns]")
    dates = df["FL_DATE"].drop_duplicates().sort_values()
    d = dates.values.astype("datetime64[ns]")
    diffs = np.abs((d[:, None] - hol[None, :]).astype("timedelta64[D]").astype(int))
    per_date = pd.Series(np.minimum(diffs.min(axis=1), 14), index=dates.values)  # cap at 2 weeks
    df["DAYS_TO_HOLIDAY"] = df["FL_DATE"].map(per_date).astype(int)

    df["LOG_DISTANCE"] = np.log1p(df["DISTANCE"])
    # scheduled clock duration (local times, so includes time-zone shift);
    # negative values mean the flight arrives after midnight
    dur = arr_min - dep_min
    df["OVERNIGHT"] = (dur < 0).astype(int)
    df["SCHED_DURATION"] = np.where(dur < 0, dur + 1440, dur)

    df["ROUTE"] = df["ORIGIN"] + "_" + df["DEST"]

    # congestion: how busy is the airport in the scheduled hour / day?
    df["ORIGIN_HOURLY_DEPS"] = df.groupby(["FL_DATE", "ORIGIN", "DEP_HOUR"])["ORIGIN"].transform("size")
    df["DEST_HOURLY_ARRS"] = df.groupby(["FL_DATE", "DEST", "ARR_HOUR"])["DEST"].transform("size")
    df["ORIGIN_DAILY_DEPS"] = df.groupby(["FL_DATE", "ORIGIN"])["ORIGIN"].transform("size")
    for c in ["ORIGIN_HOURLY_DEPS", "DEST_HOURLY_ARRS", "ORIGIN_DAILY_DEPS"]:
        df[f"LOG_{c}"] = np.log1p(df[c])
    carrier_day = df.groupby(["FL_DATE", "ORIGIN", "OP_UNIQUE_CARRIER"])["ORIGIN"].transform("size")
    df["CARRIER_DAILY_SHARE"] = carrier_day / df["ORIGIN_DAILY_DEPS"]
    return df


def join_airports(df: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Left-join airport metadata twice: once for ORIGIN, once for DEST."""
    cols = ["IATA", "LAT", "LON", "ELEV", "STATE", "IS_LARGE"]
    out = df.merge(airports[cols].add_prefix("ORIGIN_"), how="left",
                   left_on="ORIGIN", right_on="ORIGIN_IATA", validate="many_to_one")
    out = out.merge(airports[cols].add_prefix("DEST_"), how="left",
                    left_on="DEST", right_on="DEST_IATA", validate="many_to_one")
    assert len(out) == len(df), "join changed the number of rows"
    return out.drop(columns=["ORIGIN_IATA", "DEST_IATA"])


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Previous-day disruption rates (delay >= 15 min, cancelled or diverted).

    Yesterday's outcomes are known before today's departures, so these are
    legitimate predictors for a next-day forecast. They capture persistent
    conditions (multi-day storms, operational meltdowns) that the static
    schedule cannot. The first day in the data has no history -> NaN, which
    is imputed inside the model pipeline.
    """
    df = df.copy()
    df["DISRUPTED"] = ((df[TARGET] == 1) | (df["CANCELLED"] == 1) | (df["DIVERTED"] == 1)).astype(int)
    specs = [(["ORIGIN"], "ORIGIN_PREV_DAY_RATE"), (["DEST"], "DEST_PREV_DAY_RATE"),
             (["OP_UNIQUE_CARRIER"], "CARRIER_PREV_DAY_RATE"), ([], "NATIONAL_PREV_DAY_RATE")]
    for keys, name in specs:
        g = df.groupby(keys + ["FL_DATE"])["DISRUPTED"].mean().rename(name).reset_index()
        g["FL_DATE"] = g["FL_DATE"] + pd.Timedelta(days=1)   # shift: yesterday -> today
        df = df.merge(g, on=keys + ["FL_DATE"], how="left", validate="many_to_one")
    return df


def build_dataset(raw_dir: Path = RAW_DIR, airports_path: Path = EXT_DIR / "airports.csv"):
    """Full pipeline: load -> clean -> join -> features -> keep operated flights."""
    raw = load_flights(raw_dir)
    clean, log = clean_flights(raw)
    airports = load_airports(airports_path, codes=sorted(set(clean.ORIGIN) | set(clean.DEST)))
    full = join_airports(clean, airports)
    full = add_schedule_features(full)
    full = add_lag_features(full)
    operated = full.loc[(full["CANCELLED"] == 0) & (full["DIVERTED"] == 0)].copy()
    operated[TARGET] = operated[TARGET].astype(int)
    log["cancelled_or_diverted_excluded"] = int(len(full) - len(operated))
    log["modelling_rows"] = len(operated)
    return raw, full, operated, airports, log


def temporal_split(df: pd.DataFrame):
    """Train: Jan-Feb; validation: 1-14 March; test: 15-31 March."""
    train = df[df["FL_DATE"] < "2025-03-01"]
    val = df[(df["FL_DATE"] >= "2025-03-01") & (df["FL_DATE"] < "2025-03-15")]
    test = df[df["FL_DATE"] >= "2025-03-15"]
    return train, val, test
