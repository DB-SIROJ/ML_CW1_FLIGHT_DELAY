import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
from utils import TARGET, load_airports, load_sample, page_header  # noqa: E402

page_header("📊 Data Explorer", "Interactive exploratory data analysis (150k-row random sample of operated flights)")
df = load_sample()

with st.sidebar:
    st.header("Filters")
    months = st.multiselect("Month", [1, 2, 3], default=[1, 2, 3],
                            format_func=lambda m: {1: "January", 2: "February", 3: "March"}[m])
    carriers = st.multiselect("Carrier", sorted(df.CARRIER_NAME.unique()))
    top_airports = df.ORIGIN.value_counts().index.tolist()
    origins = st.multiselect("Origin airport", top_airports)
f = df[df.MONTH.isin(months)]
if carriers:
    f = f[f.CARRIER_NAME.isin(carriers)]
if origins:
    f = f[f.ORIGIN.isin(origins)]
if f.empty:
    st.warning("No flights match these filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flights (sample)", f"{len(f):,}")
c2.metric("Delay rate", f"{f[TARGET].mean():.1%}", f"{(f[TARGET].mean() - df[TARGET].mean()) * 100:+.1f} pp vs all")
c3.metric("Median arrival delay", f"{f.ARR_DELAY.median():.0f} min")
c4.metric("Mean distance", f"{f.DISTANCE.mean():,.0f} mi")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Patterns", "Distributions", "Correlations", "Map", "Statistics & raw data"])

with tab1:
    a, b = st.columns(2)
    h = f.groupby("DEP_HOUR")[TARGET].agg(["mean", "size"]).reset_index()
    a.plotly_chart(px.bar(h, x="DEP_HOUR", y="mean", hover_data=["size"], title="Delay rate by scheduled departure hour",
                          labels={"mean": "delay rate", "DEP_HOUR": "hour"}), width="stretch")
    c = f.groupby("CARRIER_NAME")[TARGET].mean().sort_values().reset_index()
    b.plotly_chart(px.bar(c, x=TARGET, y="CARRIER_NAME", orientation="h", title="Delay rate by carrier",
                          labels={TARGET: "delay rate", "CARRIER_NAME": ""}), width="stretch")
    d = f.groupby("FL_DATE")[TARGET].mean().reset_index()
    st.plotly_chart(px.line(d, x="FL_DATE", y=TARGET, markers=True, title="Daily delay rate",
                            labels={TARGET: "delay rate"}), width="stretch")
    dow = f.groupby("DAY_OF_WEEK")[TARGET].mean().rename(index=dict(enumerate("Mon Tue Wed Thu Fri Sat Sun".split()))).reset_index()
    st.plotly_chart(px.bar(dow, x="DAY_OF_WEEK", y=TARGET, title="Delay rate by weekday"), width="stretch")

with tab2:
    num = ["ARR_DELAY", "DISTANCE", "SCHED_DURATION", "ORIGIN_HOURLY_DEPS", "ORIGIN_DAILY_DEPS",
           "ORIGIN_PREV_DAY_RATE", "CARRIER_PREV_DAY_RATE", "DEP_HOUR"]
    col = st.selectbox("Variable", num)
    clip = st.checkbox("Clip display to 1st–99th percentile", value=True)
    g = f.copy()
    if clip:
        lo, hi = g[col].quantile([.01, .99])
        g = g[g[col].between(lo, hi)]
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(g, x=col, nbins=80, color=g[TARGET].map({0: "on time", 1: "delayed"}),
                                barmode="overlay", opacity=.6, title=f"Histogram of {col}"), width="stretch")
    b.plotly_chart(px.box(g, x=g[TARGET].map({0: "on time", 1: "delayed"}), y=col, title=f"{col} by class",
                          points=False), width="stretch")

with tab3:
    cols = ["ARR_DEL15", "DEP_HOUR", "DISTANCE", "SCHED_DURATION", "ORIGIN_HOURLY_DEPS", "DEST_HOURLY_ARRS",
            "ORIGIN_DAILY_DEPS", "CARRIER_DAILY_SHARE", "ORIGIN_LAT", "DEST_LAT", "ORIGIN_PREV_DAY_RATE",
            "DEST_PREV_DAY_RATE", "CARRIER_PREV_DAY_RATE", "NATIONAL_PREV_DAY_RATE"]
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    corr = f[cols].corr(method=method)
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              aspect="auto", title=f"{method.title()} correlation matrix"), width="stretch")
    x = st.selectbox("Scatter: x", cols[1:], index=9)
    agg = f.groupby(["ORIGIN", "FL_DATE"]).agg(x=(x, "mean"), rate=(TARGET, "mean"), n=(TARGET, "size")).query("n >= 10").reset_index()
    st.plotly_chart(px.scatter(agg, x="x", y="rate", opacity=.3, labels={"x": x, "rate": "airport-day delay rate"},
                               title="Airport-day scatter"), width="stretch")

with tab4:
    ap = load_airports()
    m = f.groupby("ORIGIN")[TARGET].agg(["mean", "size"]).reset_index().merge(ap, left_on="ORIGIN", right_on="IATA")
    fig = px.scatter_geo(m, lat="LAT", lon="LON", size="size", color="mean", hover_name="AIRPORT_NAME",
                         color_continuous_scale="RdYlGn_r", scope="usa", title="Origin airports: delay rate (colour) and volume (size)",
                         labels={"mean": "delay rate"})
    st.plotly_chart(fig, width="stretch")

with tab5:
    st.subheader("Summary statistics")
    s = f[["ARR_DELAY", "DISTANCE", "SCHED_DURATION", "DEP_HOUR", "ORIGIN_HOURLY_DEPS", TARGET]].describe().T
    s["variance"] = f[s.index].var()
    s["skew"] = f[s.index].skew()
    st.dataframe(s.round(3), width="stretch")
    st.subheader("Raw rows")
    st.dataframe(f.head(500), width="stretch")
    st.download_button("Download filtered sample (CSV)", f.to_csv(index=False).encode(), "flights_filtered.csv")
