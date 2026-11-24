"""Model pipelines and evaluation helpers shared by the notebook and the app."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeClassifier

from .features import HIGH_CARD_CATEGORICAL, LOW_CARD_CATEGORICAL, NUMERIC_FEATURES

RANDOM_STATE = 42


def make_preprocessor(numeric=NUMERIC_FEATURES, low=LOW_CARD_CATEGORICAL,
                      high=HIGH_CARD_CATEGORICAL) -> ColumnTransformer:
    """Imputation + scaling for numbers, one-hot for low-cardinality categories,
    cross-fitted target encoding for high-cardinality categories."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    low_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    high_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        # cross-fitting (5 folds) prevents the encoder leaking each row's own label
        ("target_enc", TargetEncoder(target_type="binary", smooth="auto",
                                     cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))),
    ])
    return ColumnTransformer([("num", numeric_pipe, list(numeric)),
                              ("low", low_pipe, list(low)),
                              ("high", high_pipe, list(high))])


def make_estimator(name: str, **params):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **params)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    if name == "Random Forest":
        return RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, **params)
    if name == "LightGBM":
        return LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params)
    raise ValueError(name)


def make_pipeline(name: str, **params) -> Pipeline:
    return Pipeline([("pre", make_preprocessor()), ("model", make_estimator(name, **params))])


def best_f1_threshold(y_true, proba) -> float:
    """Threshold that maximises F1 (chosen on the validation set, never on test)."""
    grid = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "TPR (Recall)": recall_score(y_true, pred, zero_division=0),
        "TNR (Specificity)": tn / (tn + fp) if (tn + fp) else 0.0,
        "Precision": precision_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Balanced Acc.": balanced_accuracy_score(y_true, pred),
        "Brier": brier_score_loss(y_true, proba),
        "Threshold": threshold,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }
