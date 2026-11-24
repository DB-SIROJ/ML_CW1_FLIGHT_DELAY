import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from ui import C, plain, takeaway  # noqa: E402
from utils import (ALL_FEATURES, CARRIER_NAMES, airport_label, build_feature_row, load_airports,  # noqa: E402
                   load_metadata, load_model, load_routes, load_sample, prev_day_rates)

st.title("🛫 Delay checker")
plain("Describe a flight and the final model (LightGBM, trained on 1 million January–February 2025 flights) "
      "estimates the chance that it arrives <b>15+ minutes late</b>. It also shows <b>which factors</b> pushed the risk up or down.")

model = load_model()
meta = load_metadata()
thr = meta["threshold"]
ap = load_airports().set_index("IATA")
routes = load_routes()
base_rate = load_sample()["ARR_DEL15"].mean()

GROUP_OF = {  # transformed feature -> human group
    "Time of day": ["DEP_HOUR", "ARR_HOUR", "DEP_TIME_BLOCK"],
    "Day of the week": ["DAY_OF_WEEK", "IS_WEEKEND"],
    "Yesterday's disruption": ["PREV_DAY_RATE"],
    "Airline": ["OP_UNIQUE_CARRIER", "CARRIER_DAILY_SHARE"],
    "Airport busyness": ["HOURLY_DEPS", "HOURLY_ARRS", "DAILY_DEPS"],
    "Airports & route": ["ORIGIN", "DEST", "ROUTE", "DISTANCE", "SCHED_DURATION", "OVERNIGHT"],
}


def group_contributions(X: pd.DataFrame) -> pd.Series:
    """Exact TreeSHAP contributions (log-odds) of the LightGBM model, summed into human-readable groups."""
    pre, est = model.named_steps["pre"], model.named_steps["model"]
    if not hasattr(est, "booster_"):
        return pd.Series(dtype=float)
    contrib = est.predict(pre.transform(X), pred_contrib=True)[0][:-1]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    out = {}
    for n, v in zip(names, contrib):
        g = next((g for g, keys in GROUP_OF.items() if any(k in n for k in keys)), "Other")
        out[g] = out.get(g, 0.0) + v
    return pd.Series(out).sort_values()


tab1, tab2 = st.tabs(["✈️ One flight", "📄 Many flights (CSV)"])
with tab1:
    c1, c2, c3 = st.columns(3)
    carrier = c1.selectbox("Airline", list(CARRIER_NAMES), index=3, format_func=lambda c: f"{CARRIER_NAMES[c]} ({c})")
    origins = sorted(routes.ORIGIN.unique())
    origin = c2.selectbox("From", origins, index=origins.index("ATL"), format_func=airport_label)
    dests = sorted(routes.loc[routes.ORIGIN == origin, "DEST"].unique())
    dest = c3.selectbox("To", dests, index=dests.index("LGA") if "LGA" in dests else 0, format_func=airport_label,
                        help="Only airports with direct flights from the departure airport are listed.")
    date = c1.date_input("Date", dt.date(2025, 3, 21), min_value=dt.date(2025, 1, 2), max_value=dt.date(2025, 4, 1),
                         help="Limited to the period covered by the data.")
    dep = c2.time_input("Departs (local time)", dt.time(17, 30), step=300)
    typical_dur = routes.set_index(["ORIGIN", "DEST"]).DISTANCE.get((origin, dest), 800) / 8 + 30
    default_arr = (dt.datetime.combine(date, dep) + dt.timedelta(minutes=int(typical_dur))).time()
    arr = c3.time_input("Arrives (local time)", default_arr, step=300, help="Pre-filled with a typical duration; change it if you know the schedule.")

    prev, is_actual = prev_day_rates(origin, dest, carrier, date)
    yday = pd.Timestamp(date) - pd.Timedelta(days=1)
    with st.container(border=True):
        st.markdown(f"**🌩️ How did yesterday ({yday:%d %b %Y}) go?** Looked up automatically from the data. You don't need to know this.")
        m = st.columns(4)
        m[0].metric(f"Departures from {origin}", f"{prev['origin']:.0%}", help="Share late 15+ min, cancelled or diverted")
        m[1].metric(f"Arrivals at {dest}", f"{prev['dest']:.0%}")
        m[2].metric(f"All {CARRIER_NAMES[carrier]} flights", f"{prev['carrier']:.0%}")
        m[3].metric("All U.S. flights", f"{prev['national']:.0%}")
        if not is_actual:
            st.caption("No data for that day, so typical values are shown.")
        with st.expander("What if yesterday had been different? (optional)"):
            override = st.toggle("Use my own values")
            s = st.columns(4)
            op = s[0].slider(f"{origin}", 0.0, 1.0, round(prev["origin"], 2), .01, disabled=not override)
            dp = s[1].slider(f"{dest}", 0.0, 1.0, round(prev["dest"], 2), .01, disabled=not override)
            cp = s[2].slider("Airline", 0.0, 1.0, round(prev["carrier"], 2), .01, disabled=not override)
            npv = s[3].slider("Nationwide", 0.0, 1.0, round(prev["national"], 2), .01, disabled=not override)
    if not override:
        op, dp, cp, npv = prev["origin"], prev["dest"], prev["carrier"], prev["national"]

    if st.button("🔍 Check this flight", type="primary"):
        X, dist = build_feature_row(carrier, origin, dest, date, dep.hour * 100 + dep.minute, arr.hour * 100 + arr.minute, op, dp, cp, npv)
        p = float(model.predict_proba(X)[:, 1][0])
        level = ("High", C["red"], "🔴") if p >= max(thr * 2, 0.35) else ("Elevated", C["amber"], "🟠") if p >= thr else ("Low", C["green"], "🟢")

        o, d_ = ap.loc[origin], ap.loc[dest]
        st.markdown(f"""<div class="ticket"><table style="width:100%;border:none"><tr style="border:none">
<td style="border:none;width:38%"><div class="code">{origin}</div><div class="city">{o.AIRPORT_NAME}, {o.STATE}</div><div class="city">🕔 {dep:%H:%M}</div></td>
<td style="border:none" class="mid">✈️ {CARRIER_NAMES[carrier]}<br>{pd.Timestamp(date):%a %d %b %Y}<br>{dist:,.0f} miles</td>
<td style="border:none;width:38%;text-align:right"><div class="code">{dest}</div><div class="city">{d_.AIRPORT_NAME}, {d_.STATE}</div><div class="city">🕗 {arr:%H:%M}</div></td>
</tr></table></div>""", unsafe_allow_html=True)
        st.write("")
        a, b = st.columns([1, 1])
        with a:
            st.markdown(f"""<div class="verdict" style="background:{level[1]}"><h2>{level[2]} {level[0]} risk: {p:.0%}</h2>
<p>chance of arriving 15+ minutes late</p><p>An average flight: {base_rate:.0%} · this flight: <b>{p / base_rate:.1f}×</b> that</p></div>""",
                        unsafe_allow_html=True)
            st.write("")
            verdict = "would flag this flight as **likely delayed**" if p >= thr else "would **not** flag this flight"
            st.markdown(f"The model raises an alarm above **{thr:.0%}** (chosen to balance catching delays against false alarms), so it {verdict}.")
            if p >= thr:
                st.markdown("**Tip:** an earlier departure, or avoiding a tight connection after this flight, would lower the risk.")
        with b:
            g = group_contributions(X)
            if len(g):
                g = g[g.abs() > 0.005]
                fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h",
                                       marker_color=[C["red"] if v > 0 else C["green"] for v in g.values],
                                       text=[("raises risk" if v > 0 else "lowers risk") if abs(v) >= 0.25 * g.abs().max() else "" for v in g.values],
                                       textposition="inside", insidetextanchor="middle"))
                fig.update_layout(title="Why? What pushed the risk up (red) or down (green)", height=330,
                                  xaxis_title="effect on the risk (log-odds, SHAP)", showlegend=False)
                st.plotly_chart(fig, width="stretch")
        if len(g):
            up, down = g.idxmax(), g.idxmin()
            takeaway(f"The biggest factor pushing the risk **up** is *{up}*, and the biggest pulling it **down** is *{down}*. "
                     f"For example, this flight departs at {dep:%H:%M}, and yesterday {prev['national']:.0%} of U.S. flights were disrupted.")

        hours = list(range(5, 24))
        dur = (arr.hour * 60 + arr.minute) - (dep.hour * 60 + dep.minute)
        probs = []
        for h in hours:
            am = (h * 60 + dur) % 1440
            Xh, _ = build_feature_row(carrier, origin, dest, date, h * 100, (am // 60) * 100 + am % 60, op, dp, cp, npv)
            probs.append(float(model.predict_proba(Xh)[:, 1][0]))
        w = go.Figure(go.Bar(x=[f"{h:02d}:00" for h in hours], y=probs,
                             marker_color=[C["blue"] if h != dep.hour else C["amber"] for h in hours]))
        w.add_hline(y=thr, line_dash="dash", annotation_text="alarm threshold", annotation_position="top left")
        w.update_layout(title="Same flight, different departure time (orange = your choice)", yaxis_tickformat=".0%", height=320)
        st.plotly_chart(w, width="stretch")
        best_h = hours[int(np.argmin(probs))]
        st.caption(f"Lowest risk at {best_h:02d}:00 ({min(probs):.0%}).")
        with st.expander("🔬 For the technically curious: the exact numbers sent to the model"):
            st.dataframe(X.T.rename(columns={0: "value"}).astype(str), width="stretch")

with tab2:
    st.markdown("Upload a CSV with the columns `carrier, origin, dest, date, dep_hhmm, arr_hhmm` "
                "(date as YYYY-MM-DD, times as e.g. 1730). Yesterday's disruption is looked up automatically.")
    tmpl = pd.DataFrame({"carrier": ["DL", "WN", "AA"], "origin": ["ATL", "MDW", "DFW"], "dest": ["LGA", "DEN", "ORD"],
                         "date": ["2025-03-17", "2025-03-22", "2025-03-26"], "dep_hhmm": [1730, 700, 2010], "arr_hhmm": [1945, 815, 2225]})
    st.download_button("⬇️ Download a template", tmpl.to_csv(index=False).encode(), "template.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        inp = pd.read_csv(up)
        need = {"carrier", "origin", "dest", "date", "dep_hhmm", "arr_hhmm"}
        if not need.issubset(inp.columns):
            st.error(f"Missing columns: {sorted(need - set(inp.columns))}")
        else:
            rows, bad = [], []
            for i, r in inp.iterrows():
                if r.origin not in ap.index or r.dest not in ap.index or r.carrier not in CARRIER_NAMES:
                    bad.append(i + 1); continue
                X, _ = build_feature_row(r.carrier, r.origin, r.dest, r.date, int(r.dep_hhmm), int(r.arr_hhmm))
                rows.append((i, X))
            if bad:
                st.warning(f"Skipped row(s) {bad}: unknown airline or airport code.")
            if rows:
                Xb = pd.concat([x for _, x in rows], ignore_index=True)[ALL_FEATURES]
                out = inp.loc[[i for i, _ in rows]].copy()
                out["delay_risk"] = model.predict_proba(Xb)[:, 1]
                out["verdict"] = np.where(out.delay_risk >= thr, "likely delayed", "likely on time")
                st.dataframe(out.style.format({"delay_risk": "{:.0%}"}).background_gradient(subset=["delay_risk"], cmap="RdYlGn_r"),
                             width="stretch", hide_index=True)
                st.download_button("⬇️ Download results", out.to_csv(index=False).encode(), "predictions.csv")
