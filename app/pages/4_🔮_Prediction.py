import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")
from utils import (ALL_FEATURES, CARRIER_NAMES, airport_label, build_feature_row, load_airports,  # noqa: E402
                   load_metadata, load_model, load_routes, load_sample, page_header, prev_day_rates)

page_header("🔮 Delay Prediction", "Score a flight with the final tuned model (trained on 1.02M Jan–Feb 2025 flights)")
model = load_model()
meta = load_metadata()
thr = meta["threshold"]
ap = load_airports()
df = load_sample()
routes = load_routes()  # every route flown in Jan–Mar 2025 (full data)

tab1, tab2 = st.tabs(["Single flight", "Batch (CSV upload)"])
with tab1:
    st.markdown("#### 1 · Describe the flight")
    c1, c2, c3 = st.columns(3)
    carrier = c1.selectbox("Carrier", list(CARRIER_NAMES), format_func=lambda c: f"{c} — {CARRIER_NAMES[c]}", index=3)
    origins = sorted(routes.ORIGIN.unique())
    origin = c2.selectbox("Origin airport", origins, index=origins.index("ATL"), format_func=airport_label)
    dests = sorted(routes.loc[routes.ORIGIN == origin, "DEST"].unique()) or sorted(ap.IATA)
    dest = c3.selectbox("Destination airport", dests, index=dests.index("LGA") if "LGA" in dests else 0,
                        format_func=airport_label, help="Only destinations with scheduled service from the chosen origin are listed.")
    date = c1.date_input("Flight date", dt.date(2025, 3, 21), min_value=dt.date(2025, 1, 2), max_value=dt.date(2025, 4, 1),
                         help="The model was built on Jan–Mar 2025 data, so dates are limited to that period (plus 1 April).")
    dep = c2.time_input("Scheduled departure (local time)", dt.time(17, 30), step=300)
    arr = c3.time_input("Scheduled arrival (local time)", dt.time(19, 45), step=300)

    prev, is_actual = prev_day_rates(origin, dest, carrier, date)
    yday = (pd.Timestamp(date) - pd.Timedelta(days=1)).strftime("%d %b %Y")
    st.markdown("#### 2 · Yesterday's conditions (filled in automatically)")
    st.caption(
        f"The model also looks at how disrupted the air system was the day before the flight ({yday}): the share of flights "
        "that were delayed by 15+ minutes, cancelled or diverted. Bad weather and operational meltdowns often last several days, "
        "so this is one of the strongest predictors. **You do not need to know these numbers.** They are looked up from the "
        "BTS data for the chosen date. In a live system they would come from the previous day's published flight statistics."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Departures from {origin}", f"{prev['origin']:.0%}", help=f"Share of {yday} departures from {origin} that were disrupted")
    m2.metric(f"Arrivals at {dest}", f"{prev['dest']:.0%}", help=f"Share of {yday} arrivals at {dest} that were disrupted")
    m3.metric(f"{CARRIER_NAMES[carrier]} network", f"{prev['carrier']:.0%}", help=f"Share of all {CARRIER_NAMES[carrier]} flights on {yday} that were disrupted")
    m4.metric("All U.S. flights", f"{prev['national']:.0%}", help=f"Share of all U.S. domestic flights on {yday} that were disrupted")
    if not is_actual:
        st.info("This date is outside the data, so typical values for these airports and this carrier are used.")

    with st.expander("🌩️ What-if scenario (optional): change yesterday's conditions"):
        st.caption("Move the sliders to simulate, for example, a storm on the previous day (high values) or a calm day (low values).")
        override = st.checkbox("Use my own values instead of the real ones")
        d1, d2, d3, d4 = st.columns(4)
        op = d1.slider(f"Origin {origin}", 0.0, 1.0, round(prev["origin"], 2), .01, disabled=not override)
        dp = d2.slider(f"Destination {dest}", 0.0, 1.0, round(prev["dest"], 2), .01, disabled=not override)
        cp = d3.slider("Carrier network", 0.0, 1.0, round(prev["carrier"], 2), .01, disabled=not override)
        npv = d4.slider("Nationally", 0.0, 1.0, round(prev["national"], 2), .01, disabled=not override)
    if not override:
        op, dp, cp, npv = prev["origin"], prev["dest"], prev["carrier"], prev["national"]

    go_ = st.button("Predict", type="primary")
    if go_:
        X, dist = build_feature_row(carrier, origin, dest, date, dep.hour * 100 + dep.minute, arr.hour * 100 + arr.minute,
                                    op, dp, cp, npv)
        prob = float(model.predict_proba(X)[:, 1][0])
        a, b = st.columns([1, 1])
        fig = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number={"suffix": "%"},
                                     title={"text": "Probability of arriving ≥15 min late"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#C44E52" if prob >= thr else "#4C72B0"},
                                            "threshold": {"line": {"color": "black", "width": 3}, "value": thr * 100},
                                            "steps": [{"range": [0, thr * 100], "color": "#e8f0fa"}, {"range": [thr * 100, 100], "color": "#fbe9e9"}]}))
        fig.update_layout(height=300, margin=dict(t=60, b=0))
        a.plotly_chart(fig, width="stretch")
        verdict = "⚠️ **Likely delayed**" if prob >= thr else "✅ **Likely on time**"
        base = df["ARR_DEL15"].mean()
        b.markdown(f"### {verdict}\nDecision threshold = **{thr:.2f}** (chosen on validation to maximise F1).\n\n"
                   f"This flight's risk is **{prob / base:.1f}×** the average flight's ({base:.1%}).\n\n"
                   f"**{airport_label(origin)}** → **{airport_label(dest)}**\n\nRoute distance ≈ {dist:,.0f} miles · model: {meta['final_model']}")
        with st.expander("Feature vector sent to the model"):
            st.dataframe(X.T.rename(columns={0: "value"}).astype(str), width="stretch")
        # what-if: same flight at every hour of the day
        hours = list(range(5, 24))
        probs = []
        dur = (arr.hour * 60 + arr.minute) - (dep.hour * 60 + dep.minute)
        for h in hours:
            am = (h * 60 + dur) % 1440
            Xh, _ = build_feature_row(carrier, origin, dest, date, h * 100, (am // 60) * 100 + am % 60, op, dp, cp, npv)
            probs.append(float(model.predict_proba(Xh)[:, 1][0]))
        w = go.Figure(go.Bar(x=[f"{h:02d}:00" for h in hours], y=probs))
        w.add_hline(y=thr, line_dash="dash", annotation_text="threshold")
        w.update_layout(title="What-if: same flight, different departure hour", yaxis_title="P(delay)", height=320)
        st.plotly_chart(w, width="stretch")

with tab2:
    st.markdown("Upload a CSV with columns `carrier, origin, dest, date (YYYY-MM-DD), dep_hhmm, arr_hhmm`. "
                "The previous-day disruption rates use typical values.")
    tmpl = pd.DataFrame({"carrier": ["DL", "WN", "AA"], "origin": ["ATL", "MDW", "DFW"], "dest": ["LGA", "DEN", "ORD"],
                         "date": ["2025-03-21", "2025-03-22", "2025-03-23"], "dep_hhmm": [1730, 700, 2010], "arr_hhmm": [1945, 815, 2225]})
    st.download_button("Download template", tmpl.to_csv(index=False).encode(), "template.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        inp = pd.read_csv(up)
        valid = set(ap.IATA)
        rows, errs = [], []
        for i, r in inp.iterrows():
            if r.origin not in valid or r.dest not in valid or r.carrier not in CARRIER_NAMES:
                errs.append(i); continue
            X, _ = build_feature_row(r.carrier, r.origin, r.dest, r.date, int(r.dep_hhmm), int(r.arr_hhmm))
            rows.append((i, X))
        if errs:
            st.warning(f"Skipped rows with unknown carrier/airport codes: {errs}")
        if rows:
            Xb = pd.concat([x for _, x in rows], ignore_index=True)[ALL_FEATURES]
            out = inp.loc[[i for i, _ in rows]].copy()
            out["p_delay"] = model.predict_proba(Xb)[:, 1].round(4)
            out["prediction"] = (out.p_delay >= thr).map({True: "delayed", False: "on time"})
            st.dataframe(out, width="stretch")
            st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv")
