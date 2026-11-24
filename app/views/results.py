import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from ui import C, MODEL_COLORS, plain, takeaway  # noqa: E402
from utils import MODEL_NAMES, TARGET, load_metadata, load_metric_tables  # noqa: E402

test, val, ci, split, imp, preds = load_metric_tables()
meta = load_metadata()
best = meta["final_model"]
base = test.loc["Baseline (prior)", "PR-AUC"]
r = test.loc[best]

st.title("🏆 Results & model comparison")
plain(f"All four models were tuned on 1–14 March and then graded <b>once</b> on <b>328,360 flights from 15–31 March</b> "
      f"that none of them had seen. The winner is <b>{best}</b>. Below, the numbers are explained in everyday terms first, "
      "then in full technical detail.")

st.subheader(f"The final model ({best}) in plain words")
k = st.columns(4)
k[0].metric("Late flights it catches", f"{r['TPR (Recall)']:.0%}", help="Recall / true positive rate")
k[1].metric("On-time flights not flagged", f"{r['TNR (Specificity)']:.0%}", help="Specificity / true negative rate")
k[2].metric("Alarms that are right", f"{r['Precision']:.0%}", f"vs {base:.0%} if random", help="Precision")
k[3].metric("Ranking quality (ROC-AUC)", f"{r['ROC-AUC']:.2f}", "0.50 = guessing", delta_color="off")
takeaway(f"Imagine 100 flights that end up late: the model warns about **{r['TPR (Recall)'] * 100:.0f}** of them in advance. "
         f"When it raises an alarm, it is right **{r['Precision'] * 100:.0f} times in 100**, which is "
         f"{r['Precision'] / base:.1f}× better than raising alarms at random ({base * 100:.0f} in 100). "
         "Useful for planning, but far from perfect, because same-day weather is not in the data.")

st.subheader("All models side by side")
metric = st.radio("Compare by", ["PR-AUC", "ROC-AUC", "F1", "Balanced Acc.", "Brier"], horizontal=True,
                  captions=["main metric", "ranking", "balance", "fair accuracy", "lower = better"])
cmp_ = test[metric].rename("value").reset_index().rename(columns={"index": "model"})
fig = px.bar(cmp_, x="model", y="value", color="model", color_discrete_map=MODEL_COLORS, text_auto=".3f",
             title=f"{metric} on the test set")
st.plotly_chart(fig.update_layout(showlegend=False, xaxis_title=""), width="stretch")
cols = ["TPR (Recall)", "TNR (Specificity)", "ROC-AUC", "PR-AUC", "Precision", "F1", "Balanced Acc.", "Brier", "Threshold"]
st.dataframe(test[cols].style.format("{:.3f}").highlight_max(subset=cols[2:7], color="#D6F0DF")
             .highlight_min(subset=["Brier"], color="#D6F0DF"), width="stretch")
with st.expander("Why was LightGBM chosen when Logistic Regression has a slightly higher ROC-AUC?"):
    st.markdown(f"""
- **PR-AUC** is the main metric because late flights are the minority (22 %). It focuses on how well we find *them*. LightGBM is best ({test.loc['LightGBM', 'PR-AUC']:.3f}).
- The ROC-AUC values of LightGBM, Random Forest and Logistic Regression differ by **less than 0.004**, and their confidence intervals overlap almost completely. That is a tie.
- LightGBM's probabilities are **the most trustworthy** (Brier {test.loc['LightGBM', 'Brier']:.3f}, the only model better than the baseline's {test.loc['Baseline (prior)', 'Brier']:.3f}).
  The others were trained with extra weight on late flights, which inflates their probabilities.
- It also gives the fewest false alarms (highest precision and specificity).
""")
a, b = st.columns(2)
a.markdown("**95 % confidence intervals** (bootstrap, resampling whole days)")
a.dataframe(ci, width="stretch")
b.markdown("**Honest vs naive testing** (final model)")
b.dataframe(split.style.format("{:.3f}"), width="stretch")
b.caption("Shuffling flights randomly would report a much better score, because flights from the same storm day would sit in "
          "both training and testing. Our date-based split gives the realistic number.")

st.subheader("Curves")
y = preds[TARGET].values
c1, c2 = st.columns(2)
roc, prc = go.Figure(), go.Figure()
for m in MODEL_NAMES:
    p = preds[f"p_{m}"].values
    fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
    roc.add_scatter(x=fpr, y=tpr, name=m, line=dict(color=MODEL_COLORS[m]))
    prc.add_scatter(x=rc, y=pr, name=m, line=dict(color=MODEL_COLORS[m]))
roc.add_scatter(x=[0, 1], y=[0, 1], name="guessing", line=dict(dash="dash", color="grey"))
prc.add_hline(y=y.mean(), line_dash="dash", annotation_text="guessing")
roc.update_layout(title="ROC: higher and further left is better", xaxis_title="On-time flights wrongly flagged", yaxis_title="Late flights caught")
prc.update_layout(title="Precision-recall: higher is better", xaxis_title="Late flights caught (recall)", yaxis_title="Alarms that are right (precision)")
c1.plotly_chart(roc, width="stretch"); c2.plotly_chart(prc, width="stretch")

st.subheader("🎚️ Try a different alarm threshold")
plain("The model outputs a probability. We still have to decide <i>above which probability</i> to raise an alarm. "
      "A low threshold catches more delays but raises more false alarms. A high one is the opposite. Move the slider.", title="What is a threshold?")
m = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(best))
t = st.slider("Alarm threshold", 0.05, 0.80, float(meta["thresholds"][m]), 0.01, format="%.2f")
p = preds[f"p_{m}"].values; pred = (p >= t).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
x1, x2 = st.columns([1, 1])
cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actually on time", "Actually late"], columns=["Predicted on time", "Predicted late"])
x1.plotly_chart(px.imshow(cm, text_auto=",", color_continuous_scale="Blues", title="Confusion matrix (80,000 test flights)")
                .update_layout(coloraxis_showscale=False), width="stretch")
flag = pred.mean() * 100; rec = tp / (tp + fn) * 100; prec = tp / max(tp + fp, 1) * 100
with x2:
    st.markdown(f"""
#### At a threshold of {t:.2f}:
- the model flags **{flag:.0f} of every 100 flights**,
- it catches **{rec:.0f} of every 100** late flights,
- **{prec:.0f} of every 100** alarms are correct,
- **{fn:,}** late flights slip through unflagged, and **{fp:,}** on-time flights get a false alarm.
""")
    st.caption("An airline would choose the threshold based on cost: is a missed delay or an unnecessary spare crew more expensive?")

st.subheader("What does the final model rely on?")
labels = {"DEP_HOUR_SIN": "Departure hour", "NATIONAL_PREV_DAY_RATE": "Yesterday: nationwide disruption",
          "CARRIER_PREV_DAY_RATE": "Yesterday: airline disruption", "ORIGIN_PREV_DAY_RATE": "Yesterday: origin airport disruption",
          "DEST_PREV_DAY_RATE": "Yesterday: destination airport disruption", "DAY_OF_WEEK": "Day of week",
          "ARR_HOUR_SIN": "Arrival hour", "OP_UNIQUE_CARRIER": "Airline", "ORIGIN_LON": "Origin longitude (east–west)",
          "DEST_LON": "Destination longitude", "DEP_HOUR_COS": "Departure hour (2nd component)", "ROUTE": "Route",
          "ORIGIN": "Origin airport", "DEST": "Destination airport", "ARR_HOUR_COS": "Arrival hour (2nd component)"}
top = imp.head(12)[::-1]
fig = px.bar(x=top.values, y=[labels.get(i, i) for i in top.index], orientation="h", color_discrete_sequence=[C["blue"]],
             labels={"x": "How much worse the model gets when this is scrambled", "y": ""},
             title="Permutation importance (drop in PR-AUC)")
st.plotly_chart(fig.update_layout(height=450), width="stretch")
takeaway("Two things dominate: **what time the flight leaves** and **how disrupted the system was yesterday**. "
         "Airline, airports and route come next.")

st.subheader("Where does it work better or worse?")
seg = st.radio("Split by", ["Airline", "Time of day"], horizontal=True)
key = "OP_UNIQUE_CARRIER" if seg == "Airline" else "DEP_TIME_BLOCK"
rows = [{seg: k_, "ROC-AUC": roc_auc_score(s[TARGET], s[f"p_{best}"]), "flights": len(s), "share late": s[TARGET].mean()}
        for k_, s in preds.groupby(key) if s[TARGET].nunique() > 1]
sd = pd.DataFrame(rows).sort_values("ROC-AUC")
st.plotly_chart(px.bar(sd, x=seg, y="ROC-AUC", hover_data=["flights", "share late"], range_y=[0.5, 0.8],
                       color_discrete_sequence=[C["navy"]], title=f"Ranking quality by {seg.lower()}")
                .add_hline(y=r["ROC-AUC"], line_dash="dash", annotation_text="overall"), width="stretch")
daily = preds.groupby("FL_DATE").agg(actual=(TARGET, "mean"), predicted=(f"p_{best}", "mean")).reset_index()
st.plotly_chart(px.line(daily, x="FL_DATE", y=["actual", "predicted"], markers=True,
                        labels={"value": "share late", "FL_DATE": "", "variable": ""},
                        title="Each test day: actual share late vs model's average prediction").update_layout(yaxis_tickformat=".0%"),
                width="stretch")
takeaway("The model follows the ups and downs, but **one day late**: it only learns about a storm from yesterday's numbers. "
         "Adding weather forecasts is the most obvious next step.")
