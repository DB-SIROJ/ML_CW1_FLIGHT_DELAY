# ✈️ Predicting U.S. Flight Arrival Delays (Q1 2025)

End-to-end machine-learning project for **6COSC017C-n Machine Learning and Data Analytics (CW1)**.

Using only information available **before departure**, the models estimate the probability that a U.S. domestic flight
arrives **≥ 15 minutes late** (the U.S. DOT definition, `ARR_DEL15`).

| | |
|---|---|
| **Data** | 1,645,503 flights, Jan–Mar 2025 (U.S. BTS On-Time Performance) + OurAirports metadata |
| **Task** | Supervised binary classification (19.7 % positive class) |
| **Models** | Logistic Regression · Decision Tree · Random Forest · LightGBM (tuned) |
| **Validation** | Temporal split: train Jan–Feb · validate 1–14 Mar · test 15–31 Mar |
| **Final model** | LightGBM: test ROC-AUC 0.672, PR-AUC 0.381 (no-skill PR-AUC = 0.218), best calibration |
| **App** | Multi-page Streamlit dashboard: exploration, preprocessing, live training, prediction, evaluation |
| **Tests** | 23 pytest tests (cleaning, leakage checks, models, saved artefacts, every app page), run by GitHub Actions |
| **Live demo** | https://flight-delay-cw1.streamlit.app |

## Results (held-out test set, 328,360 flights)

| Model | TPR (Recall) | TNR | ROC-AUC | PR-AUC | Precision | F1 | Brier |
|---|---|---|---|---|---|---|---|
| Baseline (prior) | 0.000 | 1.000 | 0.500 | 0.218 | 0.000 | 0.000 | 0.171 |
| Logistic Regression | 0.792 | 0.437 | **0.675** | 0.357 | 0.282 | **0.416** | 0.224 |
| Decision Tree | 0.786 | 0.397 | 0.648 | 0.355 | 0.266 | 0.398 | 0.243 |
| Random Forest | 0.766 | 0.465 | 0.672 | 0.369 | 0.285 | 0.416 | 0.209 |
| **LightGBM** | 0.661 | **0.574** | 0.672 | **0.381** | **0.302** | 0.414 | **0.160** |

Thresholds were chosen on the validation period to maximise F1. A random 80/20 split would report ROC-AUC 0.761 for the
same model. That figure is optimistic because flights from the same storm day end up in both train and test.

## Repository structure

```
├── app/                      # Streamlit multi-page application
│   ├── Home.py               #   entry point
│   ├── utils.py              #   cached loaders + feature builder for inference
│   └── pages/                #   Data Explorer · Preprocessing · Model Training · Prediction · Evaluation
├── data/
│   ├── raw/                  # BTS monthly CSVs (Jan, Feb, Mar 2025)
│   ├── external/airports.csv # OurAirports metadata
│   └── processed/            # 150k-row sample + airport lookup used by the app
├── models/                   # final_model.joblib, metrics, bootstrap CIs, importance, predictions sample
├── notebooks/
│   └── flight_delay_analysis.ipynb   # the full, executed analysis
├── reports/figures/          # figures exported by the notebook (used in the report)
├── scripts/build_notebook.py # generates the notebook skeleton
├── tests/                    # pytest suite (run automatically by GitHub Actions)
├── src/
│   ├── features.py           # loading, cleaning, joining, feature engineering (shared by notebook and app)
│   └── modeling.py           # preprocessing pipeline, model factory, metrics
├── Dockerfile                # full containerisation
├── requirements.txt          # pinned dependencies
└── LICENSE                   # MIT
```

## Setup

### Option A: local Python (3.11–3.13)
```bash
git clone https://github.com/DB-SIROJ/ML_CW1_FLIGHT_DELAY.git
cd ML_CW1_FLIGHT_DELAY
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run the notebook** (takes about 20 minutes on a 2-core laptop; regenerates `models/` and `reports/figures/`):
```bash
jupyter notebook notebooks/flight_delay_analysis.ipynb
# or headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/flight_delay_analysis.ipynb
```

**Run the app:**
```bash
streamlit run app/Home.py
```
Then open http://localhost:8501.

**Run the tests:**
```bash
pytest -v
```

### Option B: Docker (fully reproducible)
```bash
docker build -t flight-delay .
docker run -p 8501:8501 flight-delay                      # app
docker run --rm -v "$PWD":/app flight-delay \
  jupyter nbconvert --to notebook --execute --inplace notebooks/flight_delay_analysis.ipynb   # notebook
```

### Deploying to Streamlit Community Cloud
1. Push this repository to GitHub.
2. On https://share.streamlit.io choose **New app**, select the repository and branch `main`, and set **Main file path** to `app/Home.py`.
3. Deploy. The app needs only `data/processed/` and `models/`, not the raw CSVs.

## App pages
| Page | Content |
|---|---|
| 📊 Data Explorer | Filters (month, carrier, airport), delay patterns, distributions, correlation matrix, airport map, summary statistics |
| 🧹 Preprocessing | Cleaning log, missing values, outlier and transformation explorer, scaler comparison, encodings, temporal split |
| 🏋️ Model Training | Choose an algorithm, hyperparameters and feature groups, then train live and see validation/test metrics and curves |
| 🔮 Prediction | Pick a carrier, airports (full names), date and times. Yesterday's disruption levels are looked up automatically from the data, with an optional what-if override. Gauge, explanation, hour-of-day what-if, batch CSV scoring |
| 📈 Evaluation | Model comparison table, bootstrap CIs, ROC/PR curves, threshold explorer with confusion matrix, permutation importance, per-segment performance |

## Data sources and licences
* **U.S. Bureau of Transportation Statistics**, *Reporting Carrier On-Time Performance (1987–present)*, TranStats:
  https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ. U.S. Government work, public domain.
* **OurAirports**, `airports.csv`: https://github.com/davidmegginson/ourairports-data. Public domain.
* Carrier names: BTS `L_UNIQUE_CARRIERS` lookup table.

## Reproducibility notes
* All random processes use `random_state=42`.
* Every preprocessing step (imputation, scaling, encoding) lives inside a scikit-learn `Pipeline` that is fitted on training data only. The app loads exactly that fitted pipeline.
* The validation set is used for tuning and threshold selection. The test set is used only once, for the final comparison.

## Licence
MIT. See [LICENSE](LICENSE).
