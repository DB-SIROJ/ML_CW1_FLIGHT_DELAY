import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

st.set_page_config(page_title="Preprocessing", page_icon="🧹", layout="wide")
from utils import (ALL_FEATURES, HIGH_CARD_CATEGORICAL, LOW_CARD_CATEGORICAL, NUMERIC_FEATURES,  # noqa: E402
                   TARGET, load_metadata, load_sample, page_header)

page_header("🧹 Preprocessing & Feature Engineering", "Every transformation applied before modelling, with interactive comparisons")
df = load_sample()
meta = load_metadata()

st.subheader("1 · Data cleaning log (full dataset)")
log = pd.Series(meta["clean_log"], name="rows").to_frame()
st.dataframe(log, width="stretch")
st.markdown("""
* **Exact duplicates** are removed so that no flight gets double weight.
* **Clock validity, impossible combinations and target consistency** are checked explicitly. None were found, but the checks stay in the pipeline to guard future data.
* **Join-key error:** Palm Beach (PBI) was re-coded in OurAirports (now DJT), so it is matched via its ICAO code `KPBI`.
* **Cancelled/diverted flights** have *structurally missing* arrival delays and are excluded from the delay-prediction population.
""")

st.subheader("2 · Missing values")
miss = df[ALL_FEATURES].isna().mean().mul(100).round(2)
miss = miss[miss > 0].rename("missing %").to_frame()
c1, c2 = st.columns([1, 2])
c1.dataframe(miss, width="stretch")
strategy = c2.radio("Imputation strategy for lag features (illustration)", ["median", "mean", "zero"], horizontal=True)
col = "ORIGIN_PREV_DAY_RATE"
fill = {"median": df[col].median(), "mean": df[col].mean(), "zero": 0.0}[strategy]
c2.write(f"`{col}` is missing on an airport's first day in the data (no history). "
         f"Fill value with **{strategy}** = `{fill:.3f}`. The project uses the **median**, which is robust to the right-skewed rates. "
         "The imputer is fitted on training data only, inside the pipeline.")

st.subheader("3 · Outliers and transformations")
var = st.selectbox("Variable", ["DISTANCE", "ORIGIN_HOURLY_DEPS", "ORIGIN_DAILY_DEPS", "ARR_DELAY"])
transform = st.radio("Transform", ["none", "log1p", "IQR clip"], horizontal=True)
x = df[var].astype(float)
if transform == "log1p":
    x = np.log1p(x.clip(lower=0)) if var != "ARR_DELAY" else np.sign(x) * np.log1p(x.abs())
elif transform == "IQR clip":
    q1, q3 = x.quantile([.25, .75])
    x = x.clip(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
a, b = st.columns(2)
a.plotly_chart(px.histogram(x, nbins=80, title=f"{var} — {transform} (skew = {x.skew():.2f})"), width="stretch")
q1, q3 = df[var].quantile([.25, .75]); fence = q3 + 1.5 * (q3 - q1)
b.metric("Share above upper IQR fence", f"{(df[var] > fence).mean():.1%}", help=f"fence = {fence:.1f}")
b.markdown("**Decision in the project:** outliers are *kept* (they are genuine long flights, busy hubs and severe delays). "
           "Skewed counts and distance are compressed with `log1p`. ARR_DELAY is only used to define the binary target.")

st.subheader("4 · Scaling")
scaler_name = st.radio("Scaler", ["StandardScaler (used)", "MinMaxScaler", "RobustScaler"], horizontal=True)
cols = ["LOG_DISTANCE", "SCHED_DURATION", "LOG_ORIGIN_HOURLY_DEPS", "ORIGIN_ELEV"]
scaler = {"StandardScaler (used)": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}[scaler_name]
scaled = pd.DataFrame(scaler.fit_transform(df[cols].fillna(df[cols].median())), columns=cols)
a, b = st.columns(2)
a.plotly_chart(px.box(df[cols].melt(), x="variable", y="value", title="Before scaling", points=False), width="stretch")
b.plotly_chart(px.box(scaled.melt(), x="variable", y="value", title=f"After {scaler_name}", points=False), width="stretch")
st.caption("Logistic regression needs comparable feature scales (gradient-based solver + L2 penalty). Tree models are scale-invariant.")

st.subheader("5 · Encoding categorical variables")
card = pd.Series({c: df[c].nunique() for c in LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL}, name="unique values (sample)")
enc = pd.DataFrame({"unique values (sample)": card,
                    "encoding": ["One-hot" if c in LOW_CARD_CATEGORICAL else "Target encoding (5-fold cross-fitted)" for c in card.index]})
st.dataframe(enc, width="stretch")
st.write(f"One-hot encoding every categorical would create **≈{int(card.sum()):,} columns**. With target encoding for the "
         f"high-cardinality ones, the model sees **{len(NUMERIC_FEATURES) + int(card[LOW_CARD_CATEGORICAL].sum()) + len(HIGH_CARD_CATEGORICAL)} columns**.")
cat = st.selectbox("Inspect target encoding for", HIGH_CARD_CATEGORICAL)
te = df.groupby(cat)[TARGET].agg(["mean", "size"]).sort_values("size", ascending=False).head(25).reset_index()
st.plotly_chart(px.bar(te, x=cat, y="mean", hover_data=["size"], title=f"Raw delay rate per {cat} (top 25 by volume)"),
                width="stretch")

st.subheader("6 · Engineered features")
desc = {
    "DEP/ARR_HOUR_SIN/COS": "cyclical encoding of the scheduled hour (23:00 is next to 00:00)",
    "DAY_OF_WEEK, IS_WEEKEND": "weekly demand cycle",
    "LOG_DISTANCE, SCHED_DURATION, OVERNIGHT": "route geometry",
    "LOG_ORIGIN_HOURLY_DEPS, LOG_DEST_HOURLY_ARRS, LOG_ORIGIN_DAILY_DEPS": "scheduled congestion (known in advance)",
    "CARRIER_DAILY_SHARE": "carrier's share of the airport's daily departures (hub strength)",
    "ORIGIN/DEST LAT, LON, ELEV, IS_LARGE, STATE": "joined from OurAirports",
    "*_PREV_DAY_RATE": "yesterday's disruption share (delayed, cancelled or diverted) at origin, destination, carrier and nationally",
}
st.table(pd.Series(desc, name="meaning").to_frame())
st.warning("Excluded on purpose: `ARR_DELAY`, `CANCELLED`, `DIVERTED` (outcomes, so leakage); `DAY_OF_MONTH` (memorises storm days); "
           "`DAYS_TO_HOLIDAY` (constant in the March validation/test period).")

st.subheader("7 · Temporal split")
split = pd.DataFrame(meta["split"]).T
st.dataframe(split, width="stretch")
d = df.groupby("FL_DATE")[TARGET].mean().reset_index()
d["set"] = np.where(d.FL_DATE < "2025-03-01", "train", np.where(d.FL_DATE < "2025-03-15", "validation", "test"))
st.plotly_chart(px.bar(d, x="FL_DATE", y=TARGET, color="set", title="Daily delay rate coloured by split"), width="stretch")
st.caption("A random split would place flights from the same storm day in both train and test, which inflates the scores. "
           "The notebook measures this effect (Section 5.6).")
