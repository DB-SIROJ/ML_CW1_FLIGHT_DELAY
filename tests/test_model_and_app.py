"""Tests for the modelling helpers, the saved artefacts and every Streamlit page."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import ALL_FEATURES, TARGET
from src.modeling import best_f1_threshold, evaluate, make_pipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "processed" / "flights_sample.parquet"
MODEL = ROOT / "models" / "final_model.joblib"


@pytest.fixture(scope="module")
def sample():
    return pd.read_parquet(SAMPLE)


def test_evaluate_on_known_confusion_matrix():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    m = evaluate(y, p, threshold=0.5)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 1, 2)
    assert m["TPR (Recall)"] == pytest.approx(0.5)
    assert m["TNR (Specificity)"] == pytest.approx(2 / 3)
    assert m["Precision"] == pytest.approx(0.5)


def test_best_threshold_separates_perfectly_ranked_scores():
    y = np.array([0] * 80 + [1] * 20)
    p = np.r_[np.linspace(0.0, 0.3, 80), np.linspace(0.6, 0.9, 20)]
    t = best_f1_threshold(y, p)
    assert 0.3 < t <= 0.6
    assert evaluate(y, p, t)["F1"] == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM"])
def test_every_pipeline_trains_and_beats_chance(sample, name):
    tr = sample[sample.FL_DATE < "2025-03-01"].sample(15_000, random_state=0)
    te = sample[sample.FL_DATE >= "2025-03-01"].sample(8_000, random_state=0)
    params = {"Random Forest": {"n_estimators": 30, "max_depth": 8}, "LightGBM": {"n_estimators": 60},
              "Decision Tree": {"max_depth": 6, "min_samples_leaf": 100}}.get(name, {})
    pipe = make_pipeline(name, **params).fit(tr[ALL_FEATURES], tr[TARGET])
    p = pipe.predict_proba(te[ALL_FEATURES])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert evaluate(te[TARGET], p)["ROC-AUC"] > 0.55


def test_saved_model_handles_unseen_categories(sample):
    model = joblib.load(MODEL)
    row = sample[ALL_FEATURES].iloc[[0]].copy()
    row[["ORIGIN", "DEST", "ROUTE", "ORIGIN_STATE"]] = ["ZZZ", "YYY", "ZZZ_YYY", "QQ"]
    row["ORIGIN_PREV_DAY_RATE"] = np.nan          # missing lag -> imputed inside the pipeline
    p = model.predict_proba(row)[:, 1]
    assert 0 <= p[0] <= 1


def test_saved_model_reproduces_reported_test_auc():
    preds = pd.read_parquet(ROOT / "models" / "test_predictions_sample.parquet")
    metrics = pd.read_csv(ROOT / "models" / "test_metrics.csv", index_col=0)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(preds[TARGET], preds["p_LightGBM"])
    assert auc == pytest.approx(metrics.loc["LightGBM", "ROC-AUC"], abs=0.01)


APP_PAGES = [ROOT / "app" / "Home.py"] + sorted((ROOT / "app" / "pages").glob("*.py"))


@pytest.mark.parametrize("page", APP_PAGES, ids=lambda p: p.stem)
def test_streamlit_page_runs_without_errors(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(page), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    if "Prediction" in page.name:
        at.button[0].click().run()
        assert not at.exception
        assert any("Likely" in m.value for m in at.markdown)
    if "Training" in page.name:
        at.button[0].click().run(timeout=300)
        assert not at.exception and at.success
