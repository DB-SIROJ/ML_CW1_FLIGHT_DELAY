import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")
from utils import (ALL_FEATURES, CARRIER_NAMES, build_feature_row, load_airports, load_metadata,  # noqa: E402
                   load_model, load_sample, lookup_tables, page_header)

page_header("🔮 Delay Prediction", "Score a flight with the final tuned model (trained on 1.02M Jan–Feb 2025 flights)")
model = load_model()
meta = load_metadata()
thr = meta["threshold"]
ap = load_airports()
df = load_sample()
lk = lookup_tables()
routes = df.groupby(["ORIGIN", "DEST"]).size().reset_index()

tab1, tab2 = st.tabs(["Single flight", "Batch (CSV upload)"])
with tab1:
    with st.form("flight"):
        c1, c2, c3 = st.columns(3)
        carrier = c1.selectbox("Carrier", list(CARRIER_NAMES), format_func=lambda c: f"{c} — {CARRIER_NAMES[c]}", index=3)
        origins = sorted(routes.ORIGIN.unique())
        origin = c2.selectbox("Origin", origins, index=origins.index("ATL"))
        dests = sorted(routes.loc[routes.ORIGIN == origin, "DEST"].unique()) or sorted(ap.IATA)
        dest = c3.selectbox("Destination", dests, index=dests.index("LGA") if "LGA" in dests else 0)
        date = c1.date_input("Flight date", dt.date(2025, 3, 21))
        dep = c2.time_input("Scheduled departure (local)", dt.time(17, 30), step=300)
        arr = c3.time_input("Scheduled arrival (local)", dt.time(19, 45), step=300)
        st.markdown("**Yesterday's disruption share** (delayed ≥15 min, cancelled or diverted). Defaults are typical values; "
                    "move them to simulate a stormy or a calm day.")
        d1, d2, d3, d4 = st.columns(4)
        op = d1.slider(f"At origin {origin}", 0.0, 1.0, float(round(lk['origin_prev'].get(origin, .2), 2)), .01)
        dp = d2.slider("At destination", 0.0, 1.0, float(round(lk['dest_prev'].get(dest, .2), 2)), .01)
        cp = d3.slider("Carrier network", 0.0, 1.0, float(round(lk['carrier_prev'].get(carrier, .2), 2)), .01)
        npv = d4.slider("Nationally", 0.0, 1.0, float(round(lk['national_prev'], 2)), .01)
        go_ = st.form_submit_button("Predict", type="primary")
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
                   f"Route distance ≈ {dist:,.0f} miles · model: {meta['final_model']}")
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
