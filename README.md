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
| **Tests** | 24 pytest tests (cleaning, leakage checks, models, saved artefacts, every app page), run by GitHub Actions |
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
│   ├── Home.py               #   entry point + navigation
│   ├── ui.py                 #   shared look: colours, CSS, explanation blocks, glossary
│   ├── utils.py              #   cached loaders + feature builder for inference
│   └── views/                #   overview · explore · prep · lab · results · checker · about
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
The app is written for non-experts: every page opens with an "In plain words" box, every chart has an automatically computed takeaway, and a glossary sits in the sidebar.

| Section | Page | Content |
|---|---|---|
| Start | 🏠 Overview | The question, how the project works in 4 steps, headline findings |
| Understand the data | 🔎 Explore the data | Filters, then *When? / Which airline? / Where? / How late? / What goes together? / Numbers & raw data* tabs |
| | 🧹 How the data was prepared | The 6 preparation steps (clean, missing values, outliers, new features, encode & scale, split), each with an interactive demo |
| Build & evaluate | 🧪 Model lab | Pick an algorithm and preset (Quick / Balanced / Tuned / Custom), train live, compare runs on a leaderboard |
| | 🏆 Results & model comparison | Final model in plain words, all models side by side, confidence intervals, curves, threshold explorer, importance, per-segment performance |
| Use it | 🛫 Delay checker | Ticket-style input with full airport names. Yesterday's disruption is looked up automatically. Risk verdict, **SHAP explanation of why**, best departure time, batch CSV scoring |
| More | 📘 About | Data sources, method, limitations, ethics, glossary |

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
