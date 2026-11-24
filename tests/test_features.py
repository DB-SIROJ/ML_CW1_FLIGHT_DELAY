"""Unit tests for cleaning, joining and feature engineering (fast: synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from src.features import (ALL_FEATURES, add_lag_features, add_schedule_features, clean_flights,
                          hhmm_to_minutes, join_airports, temporal_split)


def make_raw(rows):
    cols = ["FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME",
            "ARR_DELAY", "ARR_DEL15", "CANCELLED", "DIVERTED", "DISTANCE"]
    df = pd.DataFrame(rows, columns=cols)
    df["SOURCE_FILE"] = "test"
    return df


D1, D2 = "1/1/2025 12:00:00 AM", "1/2/2025 12:00:00 AM"
BASE = [
    [D1, "AA", "ATL", "DFW", 800, 1000, -5.0, 0.0, 0.0, 0.0, 731.0],
    [D1, "AA", "ATL", "DFW", 900, 1100, 30.0, 1.0, 0.0, 0.0, 731.0],
    [D1, "DL", "DFW", "ATL", 1200, 1500, np.nan, np.nan, 1.0, 0.0, 731.0],
    [D2, "AA", "ATL", "DFW", 800, 1000, 10.0, 0.0, 0.0, 0.0, 731.0],
    [D2, "DL", "DFW", "ATL", 2330, 230, 20.0, 1.0, 0.0, 0.0, 731.0],
]
AIRPORTS = pd.DataFrame({"IATA": ["ATL", "DFW"], "LAT": [33.6, 32.9], "LON": [-84.4, -97.0],
                         "ELEV": [1026, 607], "STATE": ["GA", "TX"], "IS_LARGE": [1, 1]})


def test_hhmm_conversion():
    assert hhmm_to_minutes(pd.Series([0, 59, 100, 1435, 2400])).tolist() == [0, 59, 60, 875, 0]


def test_clean_removes_duplicates_and_bad_records():
    rows = BASE + [BASE[0],                                                    # exact duplicate
                   [D1, "AA", "ATL", "ATL", 800, 900, 0.0, 0.0, 0.0, 0.0, 10.0],   # origin == dest
                   [D1, "AA", "ATL", "DFW", 875, 1000, 0.0, 0.0, 0.0, 0.0, 731.0],  # minutes = 75
                   [D1, "AA", "ATL", "DFW", 800, 1000, np.nan, np.nan, 0.0, 0.0, 731.0]]  # unexplained NaN
    clean, log = clean_flights(make_raw(rows))
    assert log["exact_duplicates_removed"] == 1
    assert log["impossible_records_removed"] == 1
    assert log["invalid_clock_times_removed"] == 1
    assert log["unexplained_missing_target_removed"] == 1
    assert len(clean) == len(BASE)
    assert pd.api.types.is_datetime64_any_dtype(clean["FL_DATE"])


def test_target_is_recomputed_from_delay():
    rows = [[D1, "AA", "ATL", "DFW", 800, 1000, 20.0, 0.0, 0.0, 0.0, 731.0]]  # wrong flag
    clean, log = clean_flights(make_raw(rows))
    assert log["target_inconsistencies_corrected"] == 1
    assert clean["ARR_DEL15"].iloc[0] == 1


def test_join_keeps_row_count_and_fills_attributes():
    clean, _ = clean_flights(make_raw(BASE))
    joined = join_airports(clean, AIRPORTS)
    assert len(joined) == len(clean)
    assert joined[["ORIGIN_LAT", "DEST_LAT"]].notna().all().all()


def test_join_rejects_duplicate_lookup_keys():
    clean, _ = clean_flights(make_raw(BASE))
    with pytest.raises(Exception):
        join_airports(clean, pd.concat([AIRPORTS, AIRPORTS.iloc[[0]]]))


def test_schedule_features():
    clean, _ = clean_flights(make_raw(BASE))
    f = add_schedule_features(join_airports(clean, AIRPORTS))
    red_eye = f[(f.CRS_DEP_TIME == 2330)].iloc[0]
    assert red_eye["OVERNIGHT"] == 1 and red_eye["SCHED_DURATION"] == 180
    # cyclical encoding lies on the unit circle
    assert np.allclose(f.DEP_HOUR_SIN ** 2 + f.DEP_HOUR_COS ** 2, 1)
    # congestion counts the full schedule, including the cancelled flight
    atl_day1 = f[(f.ORIGIN == "ATL") & (f.FL_DATE == "2025-01-01")]
    assert (atl_day1["ORIGIN_DAILY_DEPS"] == 2).all()


def test_lag_features_use_only_the_previous_day():
    clean, _ = clean_flights(make_raw(BASE))
    f = add_lag_features(add_schedule_features(join_airports(clean, AIRPORTS)))
    day1, day2 = f[f.FL_DATE == "2025-01-01"], f[f.FL_DATE == "2025-01-02"]
    assert day1["NATIONAL_PREV_DAY_RATE"].isna().all()          # no history on the first day
    # day 1: 1 delayed + 1 cancelled out of 3 flights -> 2/3 disrupted
    assert day2["NATIONAL_PREV_DAY_RATE"].iloc[0] == pytest.approx(2 / 3)
    # ATL departures on day 1: 1 of 2 delayed
    assert day2.loc[day2.ORIGIN == "ATL", "ORIGIN_PREV_DAY_RATE"].iloc[0] == pytest.approx(0.5)
    # changing today's outcome must not change today's lag feature (no leakage)
    rows = [r.copy() for r in BASE]; rows[3][6], rows[3][7] = 300.0, 1.0
    f2 = add_lag_features(add_schedule_features(join_airports(clean_flights(make_raw(rows))[0], AIRPORTS)))
    assert np.allclose(f2.loc[f2.FL_DATE == "2025-01-02", "NATIONAL_PREV_DAY_RATE"],
                       day2["NATIONAL_PREV_DAY_RATE"])


def test_no_outcome_columns_in_features():
    for leak in ["ARR_DELAY", "ARR_DEL15", "CANCELLED", "DIVERTED", "DISRUPTED"]:
        assert leak not in ALL_FEATURES


def test_temporal_split_is_disjoint_and_ordered():
    dates = pd.date_range("2025-01-01", "2025-03-31")
    df = pd.DataFrame({"FL_DATE": dates})
    tr, va, te = temporal_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr.FL_DATE.max() < va.FL_DATE.min() and va.FL_DATE.max() < te.FL_DATE.min()
