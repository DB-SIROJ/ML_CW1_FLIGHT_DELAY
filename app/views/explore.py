import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from ui import C, plain, takeaway  # noqa: E402
from utils import TARGET, airport_label, load_airports, load_sample  # noqa: E402

st.title("🔎 Explore the data")
plain("Each row is one flight. The thing we want to predict is whether it arrived <b>15 or more minutes late</b>. "
      "Before building any model, we look for patterns: <i>when</i> are flights late, <i>which</i> airlines and "
      "<i>where</i>? Use the filters to focus on what interests you. Every chart has a one-line takeaway underneath.")

df = load_sample()
with st.expander("🎛️ Filter the data", expanded=False):
    f1, f2, f3 = st.columns(3)
    months = f1.multiselect("Month", [1, 2, 3], default=[1, 2, 3],
                            format_func=lambda m: {1: "January", 2: "February", 3: "March"}[m])
    carriers = f2.multiselect("Airline", sorted(df.CARRIER_NAME.unique()), placeholder="All airlines")
    origins = f3.multiselect("Departure airport", df.ORIGIN.value_counts().index.tolist(),
                             format_func=airport_label, placeholder="All airports")
f = df[df.MONTH.isin(months)]
if carriers:
    f = f[f.CARRIER_NAME.isin(carriers)]
if origins:
    f = f[f.ORIGIN.isin(origins)]
if len(f) < 200:
    st.warning("Too few flights match these filters. Please widen them.")
    st.stop()

overall = df[TARGET].mean()
k = st.columns(4)
k[0].metric("Flights in view", f"{len(f):,}", help="From a 150,000-flight random sample of the full data")
diff = (f[TARGET].mean() - overall) * 100
k[1].metric("Late (15+ min)", f"{f[TARGET].mean():.1%}", f"{diff:+.1f} pts vs all flights" if abs(diff) >= 0.05 else None, delta_color="inverse")
k[2].metric("Typical arrival", f"{f.ARR_DELAY.median():+.0f} min", help="Median: half of flights arrive earlier than this")
k[3].metric("Average distance", f"{f.DISTANCE.mean():,.0f} miles")

t1, t2, t3, t4, t5, t6 = st.tabs(["⏰ When?", "✈️ Which airline?", "📍 Where?", "📏 How late?", "🔗 What goes together?", "🧾 Numbers & raw data"])

with t1:
    h = f.groupby("DEP_HOUR")[TARGET].agg(rate="mean", flights="size").reset_index()
    fig = px.bar(h, x="DEP_HOUR", y="rate", color="rate", color_continuous_scale=["#CFE0F7", C["amber"], C["red"]],
                 hover_data={"flights": ":,"}, labels={"DEP_HOUR": "Scheduled departure hour", "rate": "Share late"},
                 title="Share of flights arriving late, by departure hour")
    fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    hh = h.set_index("DEP_HOUR")["rate"]
    e, v = hh.loc[5:8].mean(), hh.loc[17:20].mean()
    takeaway(f"Early flights (05–08h) are late {e:.0%} of the time; evening flights (17–20h) {v:.0%}, about {v / e:.1f}× more. "
             "One late aircraft makes every later flight it operates late too, so delays build up during the day.")
    a, b = st.columns(2)
    dow = f.groupby("DAY_OF_WEEK")[TARGET].mean().rename(index=dict(enumerate("Mon Tue Wed Thu Fri Sat Sun".split())))
    a.plotly_chart(px.bar(x=dow.index, y=dow.values, labels={"x": "", "y": "Share late"}, title="By day of the week")
                   .update_layout(yaxis_tickformat=".0%"), width="stretch")
    d = f.groupby("FL_DATE")[TARGET].mean()
    fig = px.line(x=d.index, y=d.values, markers=True, labels={"x": "", "y": "Share late"}, title="Day by day")
    for dt_, val in d.nlargest(3).items():
        fig.add_annotation(x=dt_, y=val, text=dt_.strftime("%d %b"), showarrow=True, arrowhead=2, ay=-25)
    b.plotly_chart(fig.update_layout(yaxis_tickformat=".0%"), width="stretch")
    worst = d.idxmax()
    takeaway(f"The day matters far more than the weekday: the share of late flights swings from {d.min():.0%} to {d.max():.0%} "
             f"(worst: {worst:%d %B}). Bad days cluster together, typically during winter storms.")

with t2:
    c = f.groupby("CARRIER_NAME")[TARGET].agg(rate="mean", flights="size").sort_values("rate").reset_index()
    fig = px.bar(c, x="rate", y="CARRIER_NAME", orientation="h", hover_data={"flights": ":,"},
                 color="rate", color_continuous_scale=["#CFE0F7", C["amber"], C["red"]],
                 labels={"rate": "Share late", "CARRIER_NAME": ""}, title="Share of late flights by airline")
    fig.add_vline(x=overall, line_dash="dash", annotation_text="all flights")
    st.plotly_chart(fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%", height=480), width="stretch")
    if len(c) > 1:
        takeaway(f"Most punctual: **{c.iloc[0].CARRIER_NAME}** ({c.iloc[0].rate:.0%} late). Least punctual: "
                 f"**{c.iloc[-1].CARRIER_NAME}** ({c.iloc[-1].rate:.0%}). The airline matters, but less than time of day.")

with t3:
    ap = load_airports()
    m = (f.groupby("ORIGIN")[TARGET].agg(rate="mean", flights="size").reset_index()
         .merge(ap, left_on="ORIGIN", right_on="IATA").query("flights >= 30"))
    fig = px.scatter_geo(m, lat="LAT", lon="LON", size="flights", color="rate", hover_name="AIRPORT_NAME",
                         hover_data={"rate": ":.0%", "flights": ":,", "LAT": False, "LON": False},
                         color_continuous_scale=["#2E8B57", "#F2C14E", C["red"]], scope="usa",
                         title="Departure airports: bubble size = number of flights, colour = share late")
    st.plotly_chart(fig.update_layout(height=520, coloraxis_colorbar_tickformat=".0%"), width="stretch")
    big = m.nlargest(20, "flights").sort_values("rate", ascending=False)
    takeaway(f"Among the 20 busiest airports, **{big.iloc[0].AIRPORT_NAME}** has the highest share of late departures "
             f"({big.iloc[0].rate:.0%}) and **{big.iloc[-1].AIRPORT_NAME}** the lowest ({big.iloc[-1].rate:.0%}).")
    st.dataframe(big[["IATA", "AIRPORT_NAME", "STATE", "flights", "rate"]].rename(columns={"rate": "share late"})
                 .style.format({"share late": "{:.1%}", "flights": "{:,}"}), width="stretch", hide_index=True)

with t4:
    st.markdown("**How many minutes late (or early) do flights arrive?** Negative means early.")
    d = f.ARR_DELAY.clip(-60, 180)
    fig = px.histogram(d, nbins=120, labels={"value": "Arrival delay (minutes, cut at -60 and 180)"},
                       title="Distribution of arrival delay")
    fig.add_vline(x=15, line_dash="dash", line_color=C["red"], annotation_text="15-minute line")
    st.plotly_chart(fig.update_layout(showlegend=False), width="stretch")
    takeaway(f"Most flights arrive **early** (median {f.ARR_DELAY.median():+.0f} min). A small number are extremely late "
             f"(the longest: {f.ARR_DELAY.max():,.0f} min), which pulls the average up to {f.ARR_DELAY.mean():+.1f} min. "
             "That long tail is why we predict 'late or not' instead of exact minutes.")
    var = st.selectbox("Compare a variable for on-time vs late flights",
                       ["DISTANCE", "DEP_HOUR", "ORIGIN_HOURLY_DEPS", "ORIGIN_PREV_DAY_RATE", "SCHED_DURATION"],
                       format_func=lambda x: {"DISTANCE": "Distance (miles)", "DEP_HOUR": "Departure hour",
                                              "ORIGIN_HOURLY_DEPS": "Departures scheduled in the same hour at the airport",
                                              "ORIGIN_PREV_DAY_RATE": "Airport's disruption share yesterday",
                                              "SCHED_DURATION": "Scheduled duration (minutes)"}[x])
    g = f.assign(status=f[TARGET].map({0: "On time", 1: "Late"}))
    st.plotly_chart(px.box(g, x="status", y=var, color="status", points=False,
                           color_discrete_map={"On time": C["blue"], "Late": C["amber"]}).update_layout(showlegend=False),
                    width="stretch")

with t5:
    plain("A correlation runs from −1 to +1. Values near 0 mean two things are unrelated; values near ±1 mean they move "
          "together. The first row shows how each variable relates to being late.", title="How to read this")
    cols = ["ARR_DEL15", "DEP_HOUR", "DISTANCE", "SCHED_DURATION", "ORIGIN_HOURLY_DEPS", "ORIGIN_DAILY_DEPS",
            "ORIGIN_LAT", "ORIGIN_PREV_DAY_RATE", "DEST_PREV_DAY_RATE", "CARRIER_PREV_DAY_RATE", "NATIONAL_PREV_DAY_RATE"]
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True,
                      captions=["straight-line relationships", "any steadily rising/falling relationship"])
    corr = f[cols].corr(method=method)
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
                    .update_layout(height=560), width="stretch")
    top = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    takeaway(f"The strongest link with being late is **{top.index[0]}** (r = {corr.loc[top.index[0], TARGET]:.2f}). "
             "All links are weak on their own, which tells us delays depend on many small factors at once. "
             "That is why we use machine learning, which combines them.")
    agg = (f.groupby(["ORIGIN", "FL_DATE"]).agg(yesterday=("ORIGIN_PREV_DAY_RATE", "first"), today=(TARGET, "mean"),
                                                n=(TARGET, "size")).query("n >= 15").dropna())
    st.plotly_chart(px.scatter(agg, x="yesterday", y="today", opacity=.35, labels={
        "yesterday": "Airport's disruption share yesterday", "today": "Share of today's departures late"},
        title="Each dot = one airport on one day").update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%"),
        width="stretch")

with t6:
    st.markdown("**Summary statistics** (mean, spread and shape of each numeric variable)")
    num = ["ARR_DELAY", "DISTANCE", "SCHED_DURATION", "DEP_HOUR", "ORIGIN_HOURLY_DEPS", "ORIGIN_PREV_DAY_RATE", TARGET]
    s = f[num].describe().T
    s["variance"] = f[num].var()
    s["skewness"] = f[num].skew()
    st.dataframe(s.round(3), width="stretch")
    st.caption("Skewness far above 0 (e.g. arrival delay) means a long right tail: a few very large values.")
    st.markdown("**Raw rows**")
    show = ["FL_DATE", "CARRIER_NAME", "ORIGIN", "DEST", "DEP_HOUR", "DISTANCE", "ARR_DELAY", TARGET]
    st.dataframe(f[show].head(300), width="stretch", hide_index=True)
    st.download_button("⬇️ Download filtered rows (CSV)", f[show].to_csv(index=False).encode(), "flights_filtered.csv")
