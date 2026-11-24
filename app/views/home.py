import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st  # noqa: E402

from ui import card, hero, plain, takeaway  # noqa: E402
from utils import TARGET, load_metadata, load_metric_tables, load_sample  # noqa: E402

df = load_sample()
meta = load_metadata()
test, *_ = load_metric_tables()
best = meta["final_model"]

hero("Will my flight be late?",
     "A machine-learning study of 1.6 million U.S. domestic flights (January to March 2025) that estimates, "
     "before take-off, the chance that a flight arrives at least 15 minutes late.",
     tag="MACHINE LEARNING & DATA ANALYTICS · COURSEWORK 1")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Flights studied", "1.6 million", help="1,611,045 flights that actually operated, after cleaning")
k2.metric("Arrive 15+ min late", f"{df[TARGET].mean():.0%}", help="Roughly 1 flight in 5")
k3.metric("Airlines · airports", "14 · 333")
k4.metric("Best model", best, help=f"Test ROC-AUC {test.loc[best, 'ROC-AUC']:.2f}, PR-AUC {test.loc[best, 'PR-AUC']:.2f}")

plain(
    "Flight delays cost passengers time and airlines money (crew overtime, missed connections, compensation). "
    "If an airline knew in advance which flights were most at risk, it could add a spare aircraft, rebook tight "
    "connections or warn passengers. This app shows the whole journey: we start with raw government flight records, "
    "clean them, look for patterns, teach four different algorithms to recognise risky flights and test them honestly "
    "on flights they have never seen. At the end you can try the final model yourself.",
    title="Why this project?",
)

st.subheader("How it works")
cols = st.columns(4)
steps = [
    ("📥", "1 · Collect", "Three monthly files from the U.S. Bureau of Transportation Statistics, joined with an airport database."),
    ("🧹", "2 · Clean & enrich", "Remove duplicates, fix a mismatched airport code, then add useful information: time of day, airport busyness, yesterday's disruption."),
    ("🧠", "3 · Learn", "Train and tune Logistic Regression, a Decision Tree, a Random Forest and LightGBM on January–February flights."),
    ("🎯", "4 · Test & use", "Grade every model on 15–31 March flights it has never seen, then use the best one in the Delay checker."),
]
for c, (i, t, x) in zip(cols, steps):
    with c:
        card(i, t, x)

st.subheader("What you can do here")
a, b, c = st.columns(3)
with a:
    card("🔎", "Explore the data", "When, where and with which airlines delays happen, with a short takeaway under every chart.")
    st.page_link("views/explore.py", label="Open the data explorer →")
with b:
    card("🧪", "Train a model yourself", "Pick an algorithm, move a few sliders, press train and compare your runs on a leaderboard.")
    st.page_link("views/lab.py", label="Open the model lab →")
with c:
    card("🛫", "Check a flight", "Choose an airline, route, date and time. See the delay risk and the reasons behind it.")
    st.page_link("views/checker.py", label="Open the delay checker →")

st.subheader("Headline findings")
h = df.groupby("DEP_HOUR")[TARGET].mean()
early, evening = h.loc[5:8].mean(), h.loc[17:20].mean()
takeaway(f"**Time of day matters most.** Flights departing 17:00–20:59 are late {evening:.0%} of the time, "
         f"about {evening / early:.1f}× more often than early-morning departures ({early:.0%}). Delays snowball through the day.")
q = df["ORIGIN_PREV_DAY_RATE"].quantile([.25, .75])
calm = df.loc[df.ORIGIN_PREV_DAY_RATE <= q.iloc[0], TARGET].mean()
bad = df.loc[df.ORIGIN_PREV_DAY_RATE >= q.iloc[1], TARGET].mean()
takeaway(f"**Bad days come in a row.** When the departure airport had a rough day yesterday (top 25 %), today's flights are "
         f"late {bad:.0%} of the time, compared with {calm:.0%} after a calm day.")
takeaway(f"**Delays are only partly predictable from the timetable.** The best model ranks a random late flight above a random "
         f"on-time flight {test.loc[best, 'ROC-AUC']:.0%} of the time (50 % would be guessing). Most of the rest is same-day "
         "weather, which is not in the data. See *Results* for an honest discussion.")
