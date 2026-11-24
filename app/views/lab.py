import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.features import LAG_FEATURES  # noqa: E402
from src.modeling import best_f1_threshold, evaluate, make_estimator, make_preprocessor  # noqa: E402
from ui import MODEL_COLORS, plain, takeaway  # noqa: E402
from utils import HIGH_CARD_CATEGORICAL, LOW_CARD_CATEGORICAL, NUMERIC_FEATURES, TARGET, load_metadata, load_sample  # noqa: E402

st.title("🧪 Model lab")
plain("Here you can <b>train a model yourself</b>, live, on a sample of the data. Pick an algorithm, choose how much effort "
      "to spend, press <b>Train</b>, and see how well it predicts flights from <b>15–31 March</b>, which it has never seen. "
      "Every run is added to the leaderboard at the bottom so you can compare.")
df = load_sample()
meta = load_metadata()

ALGOS = {
    "Logistic Regression": ("📏", "Adds up the effect of each feature in a straight-line way. Simple, fast and easy to explain."),
    "Decision Tree": ("🌳", "Asks a series of yes/no questions (e.g. 'departs after 5 pm?'). Easy to read, but can memorise noise."),
    "Random Forest": ("🌲🌲", "Hundreds of trees, each trained on a random slice of the data, then averaged. More stable than one tree."),
    "LightGBM": ("⚡", "Builds trees one after another, each fixing the previous ones' mistakes. Usually the strongest on tables."),
}
GROUPS = {
    "⏰ Time & weekday": ["DEP_HOUR_SIN", "DEP_HOUR_COS", "ARR_HOUR_SIN", "ARR_HOUR_COS", "DAY_OF_WEEK", "IS_WEEKEND"],
    "🛣️ Route length": ["LOG_DISTANCE", "SCHED_DURATION", "OVERNIGHT"],
    "🚦 Airport busyness": ["LOG_ORIGIN_HOURLY_DEPS", "LOG_DEST_HOURLY_ARRS", "LOG_ORIGIN_DAILY_DEPS", "CARRIER_DAILY_SHARE"],
    "🗺️ Airport location & size": ["ORIGIN_LAT", "ORIGIN_LON", "ORIGIN_ELEV", "DEST_LAT", "DEST_LON", "DEST_ELEV", "ORIGIN_IS_LARGE", "DEST_IS_LARGE"],
    "🌩️ Yesterday's disruption": LAG_FEATURES,
}
PRESETS = {
    "Logistic Regression": {"Quick": {"C": 1.0}, "Balanced": {"C": 0.1, "class_weight": "balanced"}},
    "Decision Tree": {"Quick": {"max_depth": 5, "min_samples_leaf": 500}, "Balanced": {"max_depth": 10, "min_samples_leaf": 300}},
    "Random Forest": {"Quick": {"n_estimators": 50, "max_depth": 10, "min_samples_leaf": 50, "n_jobs": 1},
                      "Balanced": {"n_estimators": 120, "max_depth": 14, "min_samples_leaf": 30, "max_features": 0.3, "n_jobs": 1}},
    "LightGBM": {"Quick": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31},
                 "Balanced": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 300}},
}

st.subheader("1 · Choose an algorithm")
name = st.radio("Algorithm", list(ALGOS), index=3, horizontal=True, label_visibility="collapsed",
                format_func=lambda n: f"{ALGOS[n][0]} {n}")
st.info(f"**{name}:** {ALGOS[name][1]}")

st.subheader("2 · Choose the settings")
a, b = st.columns([1, 2])
preset = a.radio("Preset", ["Quick", "Balanced", "Tuned (from the notebook)", "Custom"],
                 captions=["fastest, rough", "good default", "the best settings found by the search", "set every knob yourself"])
n_train = a.select_slider("Training flights (Jan–Feb sample)", [10_000, 25_000, 50_000, 75_000, 100_000], value=50_000)
with b:
    chosen = st.multiselect("Information the model may use", list(GROUPS), default=list(GROUPS))
    use_cats = st.toggle("Also use airline, airports and route", value=True)
    st.caption("Tip: remove '🌩️ Yesterday's disruption' and train again to see how much it matters.")

if preset == "Tuned (from the notebook)":
    params = {k: v for k, v in meta["best_params"][name].items()}
    if name == "Random Forest":
        params["n_jobs"] = 1
elif preset == "Custom":
    params = {}
    with st.expander("🔧 Hyperparameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        if name == "Logistic Regression":
            params["C"] = c1.select_slider("C: lower = simpler model", [0.001, 0.01, 0.1, 1.0, 10.0], value=1.0)
            params["class_weight"] = c2.selectbox("Give late flights more weight?", [None, "balanced"])
        elif name == "Decision Tree":
            params["max_depth"] = c1.slider("Max depth (number of questions)", 2, 25, 8)
            params["min_samples_leaf"] = c2.select_slider("Min flights per leaf", [1, 10, 50, 100, 300, 1000], value=300)
            params["criterion"] = c3.selectbox("Split criterion", ["gini", "entropy"])
        elif name == "Random Forest":
            params["n_estimators"] = c1.slider("Number of trees", 20, 300, 100, 20)
            params["max_depth"] = c2.slider("Max depth", 4, 30, 14)
            params["min_samples_leaf"] = c3.select_slider("Min flights per leaf", [1, 10, 20, 50, 100], value=30)
            params["max_features"] = c1.selectbox("Features per split", ["sqrt", 0.3, 0.5])
            params["n_jobs"] = 1
        else:
            params["n_estimators"] = c1.slider("Number of trees", 50, 1000, 300, 50)
            params["learning_rate"] = c2.select_slider("Learning rate", [0.01, 0.02, 0.05, 0.1, 0.2], value=0.05)
            params["num_leaves"] = c3.slider("Leaves per tree", 7, 255, 31)
            params["min_child_samples"] = c1.select_slider("Min flights per leaf", [20, 50, 100, 300, 1000], value=300)
else:
    params = dict(PRESETS[name][preset])
with st.expander("Show the exact settings"):
    st.json(params)

numeric = [f for g in chosen for f in GROUPS[g]]
low = LOW_CARD_CATEGORICAL if use_cats else ["DEP_TIME_BLOCK"]
high = HIGH_CARD_CATEGORICAL if use_cats else []
feats = numeric + low + high
train = df[df.FL_DATE < "2025-03-01"].sample(n_train, random_state=0)
val = df[(df.FL_DATE >= "2025-03-01") & (df.FL_DATE < "2025-03-15")]
test = df[df.FL_DATE >= "2025-03-15"]


@st.cache_resource(show_spinner=False, max_entries=10)
def fit(name, params_t, feats_t, low_t, high_t, n_train):
    pre = make_preprocessor(numeric=[f for f in feats_t if f in NUMERIC_FEATURES], low=list(low_t), high=list(high_t))
    pipe = Pipeline([("pre", pre), ("model", make_estimator(name, **dict(params_t)))])
    t0 = time.time()
    pipe.fit(train[list(feats_t)], train[TARGET])
    return pipe, time.time() - t0


st.subheader("3 · Train and see the result")
if "runs" not in st.session_state:
    st.session_state.runs = []
if st.button("🚀 Train", type="primary"):
    with st.spinner(f"Training {name} on {n_train:,} flights…"):
        pipe, secs = fit(name, tuple(sorted(params.items())), tuple(feats), tuple(low), tuple(high), n_train)
    pv, pt = pipe.predict_proba(val[feats])[:, 1], pipe.predict_proba(test[feats])[:, 1]
    thr = best_f1_threshold(val[TARGET], pv)
    mt, mv = evaluate(test[TARGET], pt, thr), evaluate(val[TARGET], pv, thr)
    st.session_state.runs.append({"#": len(st.session_state.runs) + 1, "Algorithm": name, "Preset": preset,
                                  "Info used": f"{len(chosen)}/5 groups" + (" + categories" if use_cats else ""),
                                  "Train rows": n_train, "Seconds": round(secs, 1), "ROC-AUC": mt["ROC-AUC"],
                                  "PR-AUC": mt["PR-AUC"], "Recall": mt["TPR (Recall)"], "Precision": mt["Precision"],
                                  "F1": mt["F1"], "_fpr_tpr": roc_curve(test[TARGET], pt)[:2]})
    base = test[TARGET].mean()
    m = st.columns(4)
    m[0].metric("ROC-AUC (test)", f"{mt['ROC-AUC']:.3f}", f"{mt['ROC-AUC'] - 0.5:+.3f} vs guessing")
    m[1].metric("PR-AUC (test)", f"{mt['PR-AUC']:.3f}", f"{mt['PR-AUC'] - base:+.3f} vs guessing")
    m[2].metric("Late flights caught (recall)", f"{mt['TPR (Recall)']:.0%}")
    m[3].metric("Alarms that were right (precision)", f"{mt['Precision']:.0%}", f"{mt['Precision'] - base:+.0%} vs guessing")
    takeaway(f"Trained in {secs:.1f} s. Out of every 100 flights that were really late, this model flags "
             f"**{mt['TPR (Recall)'] * 100:.0f}**. When it raises an alarm, it is right **{mt['Precision']:.0%}** of the time, "
             f"against {base:.0%} if alarms were raised at random. ROC-AUC on the validation days was {mv['ROC-AUC']:.3f}.")
    if name == "Decision Tree" and params.get("max_depth", 0) > 15 and params.get("min_samples_leaf", 1000) < 50:
        st.warning("A very deep tree with tiny leaves memorises the training flights (overfitting). Compare its test score with a forest.")
else:
    st.caption("Press **Train**. The same settings are remembered, so re-training them is instant.")

if st.session_state.runs:
    st.subheader("🏁 Leaderboard (your runs this session)")
    lb = pd.DataFrame(st.session_state.runs).drop(columns="_fpr_tpr").sort_values("PR-AUC", ascending=False)
    st.dataframe(lb.style.format({c: "{:.3f}" for c in ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"]})
                 .background_gradient(subset=["PR-AUC"], cmap="Blues"), width="stretch", hide_index=True)
    fig = go.Figure()
    for r in st.session_state.runs[-6:]:
        fig.add_scatter(x=r["_fpr_tpr"][0], y=r["_fpr_tpr"][1], name=f"#{r['#']} {r['Algorithm']} (AUC {r['ROC-AUC']:.3f})",
                        line=dict(color=MODEL_COLORS[r["Algorithm"]]))
    fig.add_scatter(x=[0, 1], y=[0, 1], name="guessing", line=dict(dash="dash", color="grey"))
    fig.update_layout(title="ROC curves of your last runs: higher and further left is better",
                      xaxis_title="Share of on-time flights wrongly flagged", yaxis_title="Share of late flights caught")
    st.plotly_chart(fig, width="stretch")
    if st.button("Clear leaderboard"):
        st.session_state.runs = []
        st.rerun()
    st.caption("The notebook trained every model on all 1.02 million January–February flights, so its scores (see Results) are a bit higher.")
