import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

st.set_page_config(page_title="Model Training", page_icon="🏋️", layout="wide")
from utils import (HIGH_CARD_CATEGORICAL, LOW_CARD_CATEGORICAL, MODEL_NAMES, NUMERIC_FEATURES,  # noqa: E402
                   TARGET, load_metadata, load_sample, page_header)

from src.features import LAG_FEATURES  # noqa: E402
from src.modeling import best_f1_threshold, evaluate, make_estimator, make_preprocessor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

page_header("🏋️ Interactive Model Training",
            "Train any of the four algorithms live on a sample, with your own hyperparameters and feature groups")
df = load_sample()
meta = load_metadata()

GROUPS = {
    "Clock & calendar": ["DEP_HOUR_SIN", "DEP_HOUR_COS", "ARR_HOUR_SIN", "ARR_HOUR_COS", "DAY_OF_WEEK", "IS_WEEKEND"],
    "Route geometry": ["LOG_DISTANCE", "SCHED_DURATION", "OVERNIGHT"],
    "Congestion": ["LOG_ORIGIN_HOURLY_DEPS", "LOG_DEST_HOURLY_ARRS", "LOG_ORIGIN_DAILY_DEPS", "CARRIER_DAILY_SHARE"],
    "Airport geography": ["ORIGIN_LAT", "ORIGIN_LON", "ORIGIN_ELEV", "DEST_LAT", "DEST_LON", "DEST_ELEV",
                          "ORIGIN_IS_LARGE", "DEST_IS_LARGE"],
    "Previous-day disruption": LAG_FEATURES,
}

with st.sidebar:
    st.header("Configuration")
    name = st.selectbox("Algorithm", MODEL_NAMES, index=3)
    n_train = st.select_slider("Training rows (from Jan–Feb sample)", [10_000, 25_000, 50_000, 75_000, 100_000], value=50_000)
    groups = st.multiselect("Numeric feature groups", list(GROUPS), default=list(GROUPS))
    use_cats = st.checkbox("Use categorical features (carrier, airports, route, state)", value=True)
    tuned = meta["best_params"].get(name, {})
    st.caption(f"Notebook's tuned parameters: `{tuned}`")

st.subheader(f"Hyperparameters — {name}")
p = {}
c1, c2, c3 = st.columns(3)
if name == "Logistic Regression":
    p["C"] = c1.select_slider("C (inverse L2 strength)", [0.001, 0.01, 0.1, 1.0, 10.0], value=1.0)
    p["class_weight"] = c2.selectbox("class_weight", [None, "balanced"])
elif name == "Decision Tree":
    p["max_depth"] = c1.slider("max_depth", 2, 25, 10)
    p["min_samples_leaf"] = c2.select_slider("min_samples_leaf", [1, 10, 50, 100, 200, 500, 1000], value=200)
    p["criterion"] = c3.selectbox("criterion", ["gini", "entropy"])
elif name == "Random Forest":
    p["n_estimators"] = c1.slider("n_estimators", 20, 300, 100, step=20)
    p["max_depth"] = c2.slider("max_depth", 4, 30, 14)
    p["min_samples_leaf"] = c3.select_slider("min_samples_leaf", [1, 10, 20, 50, 100, 200], value=50)
    p["max_features"] = c1.selectbox("max_features", ["sqrt", 0.3, 0.5])
    p["class_weight"] = c2.selectbox("class_weight", [None, "balanced_subsample"])
else:
    p["n_estimators"] = c1.slider("n_estimators", 50, 1000, 300, step=50)
    p["learning_rate"] = c2.select_slider("learning_rate", [0.01, 0.02, 0.05, 0.1, 0.2], value=0.05)
    p["num_leaves"] = c3.slider("num_leaves", 7, 255, 31)
    p["min_child_samples"] = c1.select_slider("min_child_samples", [20, 50, 100, 200, 500, 1000], value=200)
    p["subsample"] = c2.slider("subsample", 0.5, 1.0, 0.8); p["subsample_freq"] = 1
    p["colsample_bytree"] = c3.slider("colsample_bytree", 0.3, 1.0, 0.8)

numeric = [f for g in groups for f in GROUPS[g]]
low = LOW_CARD_CATEGORICAL if use_cats else ["DEP_TIME_BLOCK"]
high = HIGH_CARD_CATEGORICAL if use_cats else []
if not numeric and not use_cats:
    st.error("Select at least one feature group."); st.stop()

train = df[df.FL_DATE < "2025-03-01"].sample(n_train, random_state=0)
val = df[(df.FL_DATE >= "2025-03-01") & (df.FL_DATE < "2025-03-15")]
test = df[df.FL_DATE >= "2025-03-15"]
feats = numeric + low + high


@st.cache_resource(show_spinner=False, max_entries=8)
def train_model(name, params_items, feats_t, n_train, low_t, high_t):
    pre = make_preprocessor(numeric=[f for f in feats_t if f in NUMERIC_FEATURES], low=list(low_t), high=list(high_t))
    pipe = Pipeline([("pre", pre), ("model", make_estimator(name, **dict(params_items)))])
    t0 = time.time()
    pipe.fit(train[list(feats_t)], train[TARGET])
    return pipe, time.time() - t0


if st.button("🚀 Train model", type="primary"):
    with st.spinner("Training…"):
        pipe, secs = train_model(name, tuple(sorted(p.items(), key=lambda kv: kv[0])), tuple(feats), n_train, tuple(low), tuple(high))
    pv, pt = pipe.predict_proba(val[feats])[:, 1], pipe.predict_proba(test[feats])[:, 1]
    thr = best_f1_threshold(val[TARGET], pv)
    mv, mt = evaluate(val[TARGET], pv, thr), evaluate(test[TARGET], pt, thr)
    st.success(f"Trained on {n_train:,} rows with {len(feats)} input columns in {secs:.1f}s. Threshold (max-F1 on validation) = {thr:.2f}")
    keys = ["ROC-AUC", "PR-AUC", "TPR (Recall)", "TNR (Specificity)", "Precision", "F1", "Balanced Acc.", "Brier"]
    tbl = pd.DataFrame({"Validation (1–14 Mar)": [mv[k] for k in keys], "Test (15–31 Mar)": [mt[k] for k in keys]}, index=keys)
    a, b = st.columns([1, 2])
    a.dataframe(tbl.round(4), width="stretch")
    fpr, tpr, _ = roc_curve(test[TARGET], pt); pr, rc, _ = precision_recall_curve(test[TARGET], pt)
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, name=f"ROC (AUC {mt['ROC-AUC']:.3f})")
    fig.add_scatter(x=rc, y=pr, name=f"PR (AP {mt['PR-AUC']:.3f})")
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="random ROC")
    fig.update_layout(title="Test curves (ROC: FPR vs TPR · PR: recall vs precision)", xaxis_title="FPR / recall", yaxis_title="TPR / precision")
    b.plotly_chart(fig, width="stretch")
    gap = mv["ROC-AUC"] - mt["ROC-AUC"]
    st.caption(f"Train size matters: the notebook models use all 1.02M training rows. Validation–test ROC-AUC gap: {gap:+.3f}.")
    if name == "Decision Tree" and p["max_depth"] > 15 and p["min_samples_leaf"] < 50:
        st.warning("A deep tree with tiny leaves memorises noise (high variance). Compare it with Random Forest or LightGBM.")
else:
    st.info("Set the options and press **Train model**. Results are cached, so re-running the same configuration is instant.")
