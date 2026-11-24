import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st  # noqa: E402

from ui import GLOSSARY, plain  # noqa: E402

st.title("📘 About, limitations & ethics")

st.subheader("The data")
st.markdown("""
| Source | What it contains | Licence |
|---|---|---|
| [U.S. Bureau of Transportation Statistics: Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/) | Every domestic flight of the large U.S. airlines, Jan–Mar 2025 (1,645,503 records, 11 columns) | U.S. Government work, public domain |
| [OurAirports](https://github.com/davidmegginson/ourairports-data) | Location, altitude, size and state of each airport | Public domain |
| BTS carrier lookup table | Airline names for the 2-letter codes | Public domain |
""")

st.subheader("The method in one minute")
st.markdown("""
1. **Question:** will a flight arrive 15+ minutes late? This is a *supervised binary classification* problem. About 20 % of flights are late, so the classes are imbalanced.
2. **Features:** only information known **before departure**: timetable, airline, airports, airport busyness, and yesterday's disruption levels.
3. **Split by date:** learn on January–February, tune on 1–14 March, grade once on 15–31 March.
4. **Models:** Logistic Regression, Decision Tree, Random Forest, LightGBM, each tuned with a randomised search.
5. **Metrics:** PR-AUC (main), ROC-AUC, recall, specificity, precision, F1, Brier score, with bootstrap confidence intervals.
""")

st.subheader("Limitations: what this model cannot do")
plain("""
<ul>
<li><b>No weather data.</b> Same-day storms are the biggest cause of delays, so the model only notices them a day late, through "yesterday's disruption".</li>
<li><b>Only three winter months.</b> Summer thunderstorms and holiday peaks behave differently. The model should be retrained on a full year before real use.</li>
<li><b>No aircraft rotation.</b> The data has no tail numbers, so we cannot see whether the incoming aircraft is already late.</li>
<li><b>Moderate accuracy.</b> ROC-AUC ≈ 0.67: clearly better than guessing, but it should support decisions, not make them alone.</li>
</ul>""", title="Be careful")

st.subheader("Ethics and responsible use")
st.markdown("""
- **No personal data.** The records describe flights, not passengers.
- **Fairness between airlines and airports.** The model is less accurate for some airlines (see *Results → Where does it work better or worse*).
  Publishing raw "risk scores" could unfairly damage smaller carriers' or airports' reputations.
- **Automation bias.** People may over-trust a number. Probabilities are therefore always shown with their uncertainty and the reasons behind them.
- **Feedback loops.** If airlines react to the predictions (e.g. add spare aircraft), future delays change, and the model must be monitored and retrained.
""")

st.subheader("📖 Glossary")
for k, v in GLOSSARY.items():
    st.markdown(f"**{k}**: {v}")

st.subheader("Reproduce everything")
st.markdown("Code, notebook, tests and Docker setup: **[github.com/DB-SIROJ/ML_CW1_FLIGHT_DELAY](https://github.com/DB-SIROJ/ML_CW1_FLIGHT_DELAY)**")
st.code("pip install -r requirements.txt\nstreamlit run app/Home.py", language="bash")
