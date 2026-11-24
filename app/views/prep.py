import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler  # noqa: E402

from ui import C, plain, step, takeaway  # noqa: E402
from utils import HIGH_CARD_CATEGORICAL, LOW_CARD_CATEGORICAL, TARGET, load_metadata, load_sample  # noqa: E402

st.title("🧹 How the data was prepared")
plain("Raw data is never ready for a model. Below are the six preparation steps in the order they were applied, "
      "each with <b>why</b> it matters, <b>what</b> was done, and a small interactive demo where you can try the alternatives.")
df = load_sample()
meta = load_metadata()
log = meta["clean_log"]

tabs = st.tabs(["1 · Clean", "2 · Missing values", "3 · Outliers", "4 · New features", "5 · Encode & scale", "6 · Split"])

with tabs[0]:
    step(1, "Remove errors and fix the airport join")
    a, b = st.columns([3, 2])
    with a:
        st.markdown(f"""
**Why:** wrong or duplicated records teach the model wrong patterns.

**What was checked on all {log['rows_in']:,} flights:**
- Exact duplicate rows → **{log['exact_duplicates_removed']} removed**
- Impossible clock times (e.g. 14:75) → {log['invalid_clock_times_removed']} found
- Impossible combinations (origin = destination, distance ≤ 0, cancelled *and* diverted) → {log['impossible_records_removed']} found
- "Late" flag disagreeing with the delay minutes → {log['target_inconsistencies_corrected']} found

**A real problem found when joining the two sources:** Palm Beach airport (PBI) has been renamed in the airport
database, so **18,463 flights** would have lost their airport information. It was matched through its
international code (KPBI) instead.
""")
    with b:
        funnel = pd.DataFrame({"stage": ["Raw records", "After cleaning", "Operated (not cancelled/diverted)"],
                               "rows": [log["rows_in"], log["rows_out"], log["rows_out"] - 34_455]})
        st.plotly_chart(px.funnel(funnel, x="rows", y="stage", title="Rows at each stage"), width="stretch")

with tabs[1]:
    step(2, "Deal with missing values")
    st.markdown("""
| Where values are missing | Why | What we did |
|---|---|---|
| Arrival delay of **34,455 cancelled or diverted** flights | These flights never arrived, so "arrival delay" does not exist | Left them out: predicting cancellations is a different question |
| "Yesterday's disruption" on **1 January** (and an airport's first day) | No earlier day in the data | Filled with the **median** inside the model pipeline |
""")
    col = "ORIGIN_PREV_DAY_RATE"
    choice = st.radio("Try a different way of filling the gaps:", ["Median (used)", "Mean", "Zero"], horizontal=True)
    val = {"Median (used)": df[col].median(), "Mean": df[col].mean(), "Zero": 0.0}[choice]
    filled = df[col].fillna(val)
    st.plotly_chart(px.histogram(filled, nbins=60, title=f"Yesterday's disruption share after filling with {choice.split()[0].lower()} = {val:.2f}",
                                 labels={"value": "share disrupted"}).update_layout(showlegend=False), width="stretch")
    takeaway("The median is not pulled up by a few very bad days, so it represents a 'normal' day better than the mean. "
             "Zero would wrongly pretend that nothing went wrong.")

with tabs[2]:
    step(3, "Outliers: keep, cap or transform?")
    var = st.selectbox("Variable", ["DISTANCE", "ORIGIN_HOURLY_DEPS", "ARR_DELAY"],
                       format_func=lambda x: {"DISTANCE": "Distance (miles)", "ORIGIN_HOURLY_DEPS": "Departures in the same hour",
                                              "ARR_DELAY": "Arrival delay (minutes)"}[x])
    how = st.radio("Treatment", ["None", "Log transform", "Cap at IQR fence"], horizontal=True)
    x = df[var].astype(float)
    if how == "Log transform":
        x = np.sign(x) * np.log1p(x.abs())
    elif how == "Cap at IQR fence":
        q1, q3 = x.quantile([.25, .75]); x = x.clip(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    a, b = st.columns([3, 2])
    a.plotly_chart(px.histogram(x, nbins=80, title=f"{how}: skewness = {x.skew():.2f}").update_layout(showlegend=False), width="stretch")
    q1, q3 = df[var].quantile([.25, .75]); fence = q3 + 1.5 * (q3 - q1)
    b.metric("Flights above the outlier fence", f"{(df[var] > fence).mean():.1%}", help=f"Fence = Q3 + 1.5×IQR = {fence:,.0f}")
    b.markdown("**Decision:** the extreme values are **real** (long Hawaii flights, busy hubs, storm delays), not mistakes, "
               "so none were deleted. Distance and congestion were **log-transformed** to shrink the long tail. "
               "Arrival delay is only used to define 'late or not', so its extremes cannot distort the model.")

with tabs[3]:
    step(4, "Create new, more useful features")
    st.markdown("The raw file has only 11 columns. These new features were built from it. **All of them are known before the flight departs.**")
    feats = pd.DataFrame([
        ("⏰", "Time of day (circular)", "Hour turned into sine/cosine so that 23:00 sits next to 00:00", "Delays build up during the day"),
        ("📅", "Day of week, weekend flag", "From the date", "Weekly travel patterns"),
        ("🛣️", "Distance (log), scheduled duration, overnight flag", "From the timetable", "Long flights can make up time in the air"),
        ("🚦", "Airport busyness", "Number of flights scheduled at the airport in the same hour and day", "Busy hours mean queues for runways"),
        ("🗺️", "Airport location, altitude, size, state", "Joined from the OurAirports database", "Northern airports get snow in winter"),
        ("🌩️", "Yesterday's disruption", "Share of yesterday's flights that were late, cancelled or diverted, at the origin, the destination, the airline and nationwide",
         "Storms and meltdowns last several days"),
    ], columns=["", "Feature", "How it is made", "Why it might help"])
    st.dataframe(feats, width="stretch", hide_index=True)
    st.warning("**Deliberately NOT used:** the actual delay minutes and the cancellation/diversion flags. These are outcomes, and "
               "using them would be *data leakage*. Also excluded: **day of month**, because the model would memorise which days "
               "had storms in Jan–Feb, and **days to the nearest holiday**, because it has the same value for every March flight.")
    h = df.groupby("DEP_HOUR")[["DEP_HOUR_SIN", "DEP_HOUR_COS"]].first().reset_index()
    st.plotly_chart(px.scatter(h, x="DEP_HOUR_SIN", y="DEP_HOUR_COS", text="DEP_HOUR", title="Why sine/cosine? The 24 hours form a circle")
                    .update_traces(textposition="top center").update_layout(height=420, xaxis_scaleanchor="y"), width="stretch")

with tabs[4]:
    step(5, "Turn categories into numbers and put numbers on the same scale")
    card = pd.Series({c: df[c].nunique() for c in LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL})
    st.markdown(f"""
**Categories → numbers.** Models only understand numbers.
- **Few categories** (14 airlines, 5 times of day) → **one-hot encoding**: one yes/no column per category.
- **Many categories** (333 airports, ≈ 5,900 routes) → **target encoding**: each value is replaced by its average delay rate.
  This is computed on *other* parts of the training data (5-fold cross-fitting), so a flight never "sees" its own answer.
  One-hot encoding here would create about **{int(card.sum()):,} columns**.
""")
    cat = st.selectbox("See the target encoding for", HIGH_CARD_CATEGORICAL)
    te = df.groupby(cat)[TARGET].agg(rate="mean", flights="size").sort_values("flights", ascending=False).head(20).reset_index()
    st.plotly_chart(px.bar(te, x=cat, y="rate", hover_data=["flights"], title=f"Average delay rate per {cat} (20 busiest)",
                           labels={"rate": "encoded value ≈ delay rate"}).update_layout(yaxis_tickformat=".0%"), width="stretch")
    st.markdown("**Scaling.** Logistic regression needs features on comparable scales. Tree models do not care either way.")
    sname = st.radio("Scaler", ["StandardScaler (used)", "MinMaxScaler", "RobustScaler"], horizontal=True)
    cols = ["LOG_DISTANCE", "SCHED_DURATION", "LOG_ORIGIN_HOURLY_DEPS", "ORIGIN_ELEV"]
    sc = {"StandardScaler (used)": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}[sname]
    scaled = pd.DataFrame(sc.fit_transform(df[cols].fillna(df[cols].median())), columns=cols)
    a, b = st.columns(2)
    a.plotly_chart(px.box(df[cols].melt(), x="variable", y="value", points=False, title="Before"), width="stretch")
    b.plotly_chart(px.box(scaled.melt(), x="variable", y="value", points=False, title=f"After {sname}"), width="stretch")

with tabs[5]:
    step(6, "Split by time: learn from the past, test on the future")
    split = pd.DataFrame(meta["split"]).T
    st.dataframe(split, width="stretch")
    d = df.groupby("FL_DATE")[TARGET].mean().reset_index()
    d["set"] = np.where(d.FL_DATE < "2025-03-01", "Training (learn)", np.where(d.FL_DATE < "2025-03-15", "Validation (tune)", "Test (final grade)"))
    st.plotly_chart(px.bar(d, x="FL_DATE", y=TARGET, color="set", labels={TARGET: "share late", "FL_DATE": ""},
                           color_discrete_map={"Training (learn)": C["blue"], "Validation (tune)": C["amber"], "Test (final grade)": C["red"]},
                           title="Which days were used for what").update_layout(yaxis_tickformat=".0%"), width="stretch")
    takeaway("A random shuffle would put flights from the same stormy day into both training and testing, so the model would "
             "look better than it really is. Splitting by date mimics real use: we always predict days we have not seen yet.")
