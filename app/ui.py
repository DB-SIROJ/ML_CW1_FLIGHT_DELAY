"""Visual language of the app: colours, CSS, reusable explanation blocks and the glossary."""
import re

import plotly.express as px
import plotly.io as pio
import streamlit as st

C = {
    "navy": "#0B2545", "blue": "#1F6FEB", "sky": "#A9C5E8", "amber": "#E8871E",
    "green": "#2E8B57", "red": "#C8442F", "grey": "#8A94A6", "bg": "#F4F7FB",
}
MODEL_COLORS = {"Logistic Regression": "#1F6FEB", "Decision Tree": "#E8871E",
                "Random Forest": "#2E8B57", "LightGBM": "#7B4FD6", "Baseline (prior)": "#8A94A6"}

# one plotly look for every chart
pio.templates["flight"] = pio.templates["plotly_white"]
pio.templates["flight"].layout.update(
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=C["navy"]),
    colorway=[C["blue"], C["amber"], C["green"], "#7B4FD6", C["red"], C["grey"]],
    title=dict(font=dict(size=16)), margin=dict(t=50, l=10, r=10, b=10),
)
pio.templates.default = "flight"
px.defaults.template = "flight"

CSS = f"""
<style>
.block-container {{padding-top: 2rem; max-width: 1250px;}}
h1, h2, h3 {{color: {C['navy']};}}
.hero {{background: linear-gradient(120deg, {C['navy']} 0%, #13406B 60%, {C['blue']} 100%);
        color: white; padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.3rem;}}
.hero h1 {{color: white; margin: 0 0 .4rem 0; font-size: 2.3rem;}}
.hero p {{color: #DCE8F7; font-size: 1.08rem; margin: 0;}}
.hero .tag {{display:inline-block; background: rgba(255,255,255,.14); padding: .2rem .7rem;
             border-radius: 999px; font-size: .8rem; margin-bottom: .7rem; letter-spacing: .03em;}}
.plain {{background: #EEF4FF; border-left: 5px solid {C['blue']}; padding: .9rem 1.1rem;
         border-radius: 10px; margin: .4rem 0 1rem 0; color: {C['navy']};}}
.plain b.t {{display:block; font-size:.78rem; text-transform: uppercase; letter-spacing:.06em;
             color: {C['blue']}; margin-bottom:.25rem;}}
.takeaway {{background: #FFF6EA; border-left: 5px solid {C['amber']}; padding: .7rem 1rem;
            border-radius: 10px; margin: .2rem 0 1.1rem 0; color: #5A3A10;}}
.card {{background: white; border: 1px solid #E3E9F2; border-radius: 14px; padding: 1.1rem 1.2rem;
        height: 100%; box-shadow: 0 1px 3px rgba(11,37,69,.05);}}
.card .ic {{font-size: 1.7rem;}}
.card h4 {{margin: .35rem 0 .3rem 0; color: {C['navy']};}}
.card p {{margin: 0; color: #4A5A70; font-size: .93rem;}}
.step {{display:flex; align-items:center; gap:.7rem; margin: .4rem 0 .6rem 0;}}
.step .n {{background:{C['blue']}; color:white; width:2rem; height:2rem; border-radius:50%;
           display:flex; align-items:center; justify-content:center; font-weight:700;}}
.step .s {{font-size:1.25rem; font-weight:700; color:{C['navy']};}}
div[data-testid="stMetric"] {{background: white; border: 1px solid #E3E9F2; border-radius: 14px;
                              padding: .8rem 1rem; box-shadow: 0 1px 3px rgba(11,37,69,.05);}}
.ticket {{background: white; border: 2px dashed #C9D6E8; border-radius: 16px; padding: 1rem 1.4rem;}}
.ticket .code {{font-size: 2.2rem; font-weight: 800; color: {C['navy']}; line-height: 1;}}
.ticket .city {{color: #4A5A70; font-size: .9rem;}}
.ticket .mid {{text-align:center; color:{C['grey']}; font-size: .9rem;}}
.verdict {{border-radius: 16px; padding: 1.2rem 1.4rem; color: white;}}
.verdict h2 {{color: white; margin: 0;}}
.verdict p {{margin: .3rem 0 0 0; color: rgba(255,255,255,.92);}}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, tag: str = ""):
    tag_html = f'<span class="tag">{tag}</span><br>' if tag else ""
    st.markdown(f'<div class="hero">{tag_html}<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def _md(text: str) -> str:
    """Tiny markdown -> HTML for text placed inside custom HTML blocks (**bold**, *italic*)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)


def plain(text: str, title: str = "In plain words"):
    text = _md(text)
    st.markdown(f'<div class="plain"><b class="t">{title}</b>{text}</div>', unsafe_allow_html=True)


def takeaway(text: str):
    text = _md(text)
    st.markdown(f'<div class="takeaway">💡 <b>Takeaway:</b> {text}</div>', unsafe_allow_html=True)


def card(icon: str, title: str, text: str):
    st.markdown(f'<div class="card"><div class="ic">{icon}</div><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True)


def step(n, title: str):
    st.markdown(f'<div class="step"><div class="n">{n}</div><div class="s">{title}</div></div>', unsafe_allow_html=True)


GLOSSARY = {
    "Delayed flight": "A flight that arrives 15 minutes or more after its scheduled time (the official U.S. DOT definition).",
    "Feature": "A piece of information the model uses to make a prediction, e.g. the departure hour or the airline.",
    "Training / test set": "The model learns from the training flights (Jan–Feb) and is graded on test flights it has never seen (15–31 Mar).",
    "Probability": "The model's estimated chance that a flight will be late, from 0 % to 100 %.",
    "Threshold": "The probability above which we call a flight 'likely delayed'. Lower thresholds raise more alarms.",
    "Recall (TPR)": "Of all flights that really were late, the share the model caught.",
    "Specificity (TNR)": "Of all flights that were on time, the share the model correctly left alone.",
    "Precision": "Of all flights the model flagged as late, the share that really were late.",
    "F1 score": "A single score that balances precision and recall (their harmonic mean).",
    "ROC-AUC": "If you pick one late and one on-time flight at random, the chance the model gives the late one a higher risk. 0.5 = coin flip, 1.0 = perfect.",
    "PR-AUC": "How precise the model stays as it tries to catch more delays. A model that guesses randomly scores the delay rate (≈ 0.22 here).",
    "Brier score": "The average squared gap between predicted probabilities and what happened. Lower means more trustworthy probabilities.",
    "Data leakage": "Accidentally giving the model information it would not have at prediction time, which makes it look better than it is.",
    "Overfitting": "When a model memorises the training data instead of learning general patterns, so it does worse on new data.",
    "SHAP value": "How much one feature pushed a specific prediction up or down compared with an average flight.",
}


def glossary_sidebar():
    with st.sidebar.expander("📖 Glossary: what do these terms mean?"):
        for k, v in GLOSSARY.items():
            st.markdown(f"**{k}**: {v}")
