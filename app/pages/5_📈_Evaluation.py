import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

st.set_page_config(page_title="Evaluation", page_icon="📈", layout="wide")
from utils import COLORS, MODEL_NAMES, TARGET, load_metadata, load_metric_tables, page_header  # noqa: E402

page_header("📈 Model Evaluation Dashboard", "All models compared on the held-out test period (15–31 March 2025, 328,360 flights)")
test, val, ci, split, imp, preds = load_metric_tables()
meta = load_metadata()
best = meta["final_model"]

k = st.columns(5)
for col, m in zip(k, ["ROC-AUC", "PR-AUC", "TPR (Recall)", "TNR (Specificity)", "F1"]):
    col.metric(f"{best}: {m}", f"{test.loc[best, m]:.3f}",
               f"{test.loc[best, m] - test.loc['Logistic Regression', m]:+.3f} vs LogReg" if best != "Logistic Regression" else None)

st.subheader("Comparison table (test set, thresholds chosen on validation)")
cols = ["TPR (Recall)", "TNR (Specificity)", "ROC-AUC", "PR-AUC", "Precision", "F1", "Balanced Acc.", "Brier", "Threshold"]
st.dataframe(test[cols].style.format("{:.4f}").highlight_max(subset=cols[:7], color="#d4edda").highlight_min(subset=["Brier"], color="#d4edda"),
             width="stretch")
c1, c2 = st.columns(2)
c1.markdown("**Bootstrap 95% confidence intervals (resampled by day)**"); c1.dataframe(ci, width="stretch")
c2.markdown("**Random vs temporal split (final model)**"); c2.dataframe(split, width="stretch")
c2.caption("A random split leaks same-day weather between train and test and overstates performance.")

st.subheader("Curves")
chosen = st.multiselect("Models", MODEL_NAMES, default=MODEL_NAMES)
y = preds[TARGET].values
a, b = st.columns(2)
roc, prc = go.Figure(), go.Figure()
for m in chosen:
    p = preds[f"p_{m}"].values
    fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
    roc.add_scatter(x=fpr, y=tpr, name=m, line=dict(color=COLORS[m]))
    prc.add_scatter(x=rc, y=pr, name=m, line=dict(color=COLORS[m]))
roc.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="random")
prc.add_hline(y=y.mean(), line_dash="dash", annotation_text="no skill")
roc.update_layout(title="ROC", xaxis_title="False positive rate", yaxis_title="True positive rate")
prc.update_layout(title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")
a.plotly_chart(roc, width="stretch"); b.plotly_chart(prc, width="stretch")

st.subheader("Threshold explorer")
m = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(best))
t = st.slider("Decision threshold", 0.05, 0.95, float(meta["thresholds"][m]), 0.01)
p = preds[f"p_{m}"].values; pred = (p >= t).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
x1, x2 = st.columns([1, 1])
cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual on time", "Actual delayed"], columns=["Pred. on time", "Pred. delayed"])
x1.plotly_chart(px.imshow(cm, text_auto=",", color_continuous_scale="Blues", title=f"Confusion matrix @ {t:.2f} (80k-row test sample)"),
                width="stretch")
rec, spec = tp / (tp + fn), tn / (tn + fp); prec = tp / (tp + fp) if tp + fp else 0
x2.metric("Recall / TPR", f"{rec:.3f}"); x2.metric("Specificity / TNR", f"{spec:.3f}"); x2.metric("Precision", f"{prec:.3f}")
x2.metric("Flights flagged", f"{pred.mean():.1%}")
x2.caption("Lower thresholds catch more delays (recall ↑) at the cost of more false alarms (precision ↓). "
           "An airline would choose the threshold according to the relative cost of a missed delay and an unnecessary intervention.")

st.subheader("What drives the final model?")
fig = px.bar(imp.head(15)[::-1], orientation="h", labels={"value": "drop in PR-AUC when shuffled", "index": ""},
             title=f"Permutation importance — {best}")
fig.update_layout(showlegend=False)
st.plotly_chart(fig, width="stretch")

st.subheader("Performance by segment")
seg = st.radio("Segment by", ["OP_UNIQUE_CARRIER", "DEP_TIME_BLOCK", "ORIGIN (top 15)"], horizontal=True)
key = "ORIGIN" if seg.startswith("ORIGIN") else seg
g = preds if key != "ORIGIN" else preds[preds.ORIGIN.isin(preds.ORIGIN.value_counts().head(15).index)]
rows = []
for k_, s in g.groupby(key):
    if s[TARGET].nunique() > 1:
        rows.append({key: k_, "ROC-AUC": roc_auc_score(s[TARGET], s[f"p_{best}"]), "delay rate": s[TARGET].mean(), "n": len(s)})
segdf = pd.DataFrame(rows).sort_values("ROC-AUC")
st.plotly_chart(px.bar(segdf, x=key, y="ROC-AUC", hover_data=["delay rate", "n"], range_y=[0.5, 0.85],
                       title=f"{best}: ROC-AUC by {key}"), width="stretch")
daily = preds.groupby("FL_DATE").agg(actual=(TARGET, "mean"), predicted=(f"p_{best}", "mean")).reset_index()
st.plotly_chart(px.line(daily, x="FL_DATE", y=["actual", "predicted"], markers=True, title="Daily delay rate: actual vs mean predicted"),
                width="stretch")
