import streamlit as st

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="wide")

from utils import CARRIER_NAMES, TARGET, load_metadata, load_metric_tables, load_sample  # noqa: E402

st.title("✈️ U.S. Flight Arrival-Delay Prediction")
st.markdown(
    "End-to-end machine-learning project on **1.6 million U.S. domestic flights (Jan–Mar 2025)** "
    "from the U.S. Bureau of Transportation Statistics. The goal is to estimate, **before departure**, "
    "the probability that a flight arrives **15 minutes or more late** (the official DOT definition)."
)

df = load_sample()
meta = load_metadata()
test, *_ = load_metric_tables()
best = meta["final_model"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Flights analysed", "1,611,045", help="Operated flights after cleaning (cancelled/diverted excluded)")
c2.metric("Delay rate (≥15 min)", f"{df[TARGET].mean():.1%}")
c3.metric("Carriers / airports", f"{len(CARRIER_NAMES)} / 333")
c4.metric("Final model", best)
c5.metric("Test ROC-AUC / PR-AUC", f"{test.loc[best, 'ROC-AUC']:.3f} / {test.loc[best, 'PR-AUC']:.3f}")

st.divider()
left, right = st.columns([3, 2])
with left:
    st.subheader("How to use this app")
    st.markdown(
        """
| Page | What you can do |
|---|---|
| **📊 Data Explorer** | Filter by carrier, airport and month; view delay patterns, distributions, correlations and an airport map |
| **🧹 Preprocessing** | See every cleaning step, missing-value treatment, outlier handling, transformations, encodings and the temporal split, and change the options interactively |
| **🏋️ Model Training** | Pick an algorithm, set its hyperparameters and feature groups, and train it live on a sample |
| **🔮 Prediction** | Describe a flight and get its delay probability from the final model, or upload a CSV for batch scoring |
| **📈 Evaluation** | Dashboard comparing all models on the held-out test set: ROC/PR curves, a confusion matrix with a threshold slider, confidence intervals, feature importance |
"""
    )
with right:
    st.subheader("Pipeline")
    st.markdown(
        """
1. **Load & join**: 3 monthly BTS files + OurAirports metadata + carrier lookup
2. **Clean**: duplicates, validity checks, join-key fix (PBI → KPBI)
3. **Engineer**: cyclical time, congestion, airport geography, previous-day disruption
4. **Split by time**: train Jan–Feb · validate 1–14 Mar · test 15–31 Mar
5. **Tune & compare**: Logistic Regression, Decision Tree, Random Forest, LightGBM
6. **Deploy**: this app (same fitted pipeline as the notebook)
"""
    )
st.info("Data: U.S. DOT Bureau of Transportation Statistics (public domain) and OurAirports (public domain). "
        "The app runs on a 150k-row random sample so that it stays fast. All models were trained on the full data in the notebook.")
