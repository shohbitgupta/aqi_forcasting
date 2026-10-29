# Next-Day AQI Forecasting for Indian Cities

Predict **tomorrow's air-quality category** (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) for 14 Indian cities, using today's AQI, the pollutants driving it, the weather and the calendar.

DataTalks.Club ML Zoomcamp midterm project. Full problem statement: [`problem_statement.pdf`](problem_statement.pdf).

## Why it matters

Air-quality dashboards in India mostly report what the air is like *right now*. By the time the AQI reaches "Severe", people have already been exposed. A next-day forecast lets citizens plan outdoor activity, schools decide on closures, hospitals prepare, and authorities start pollution controls (Delhi's GRAP) **before** the peak rather than after.

The task is a **multi-class classification** problem. The target is the CPCB category of the next day's AQI:

| AQI | 0–50 | 51–100 | 101–200 | 201–300 | 301–400 | 401–500 |
|---|---|---|---|---|---|---|
| Category | Good | Satisfactory | Moderate | Poor | Very Poor | Severe |

### How the forecast will be used

Every evening, once the day's CPCB bulletin and weather are known, a web service takes one city's day (today's AQI, pollutants and weather, as JSON) and returns:

- the **most likely category for tomorrow**;
- the **chance of a Severe day** and of a **Dangerous day** (Very Poor or Severe);
- an **alert flag**, raised when that chance passes a level each user can set: a hospital may want earlier warnings than a public app.

**What counts as success:** beating the simple forecast "tomorrow = today", which is already right on 71% of days (2023).

## Data

One row per city per day, 14 cities, May 2015 to December 2025 (51,072 city-days).

| Region | Cities |
|---|---|
| Delhi NCR | Delhi, Noida, Greater Noida, Ghaziabad, Gurugram, Faridabad |
| Indo-Gangetic Plain | Lucknow, Kanpur, Patna |
| Other metros (cleaner air, for contrast) | Mumbai, Kolkata, Chennai, Bengaluru, Hyderabad |

Each row holds the day's AQI value and category, the prominent pollutants, the number of monitoring stations, and daily weather: temperature (mean, max, min, range), humidity, dew point, rain, surface pressure, cloud cover, and wind speed and direction.

All data is committed, so the notebooks run straight after cloning. The scripts in `code/` rebuild it from the original sources.

### Data sources and licences

| Data | Source | Licence |
|---|---|---|
| Daily AQI bulletins of the Central Pollution Control Board (CPCB) | Compiled by Urban Emissions in [urbanemissionsinfo/AQI_bulletins](https://github.com/urbanemissionsinfo/AQI_bulletins) | [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html) |
| Historical hourly weather, aggregated to daily values | [Open-Meteo](https://open-meteo.com/) Historical Weather API | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Weather data by [Open-Meteo.com](https://open-meteo.com/). The AQI files in `data/` (`data/raw/aqi_bulletins.csv`, `data/processed/AllIndiaBulletinsMaster2025_openrefined.csv` and the files derived from them) are redistributed from the Urban Emissions repository under its GPL-3.0 licence.

## Project structure

```
aqi_forcasting/
├── code/                       # data pipeline
│   ├── cities.py               # the 14 cities, their coordinates and the date range
│   ├── prepare_aqi.py          # step 1: download and clean the AQI bulletins
│   ├── download_weather.py     # step 2: download weather from Open-Meteo
│   └── merge_data.py           # step 3: merge into one dataset with a full daily calendar
├── data/
│   ├── raw/                    # downloaded files (AQI bulletins, weather per city)
│   └── processed/              # aqi_weather_2015_2025.csv is the modelling dataset
├── notebooks/                  # understanding the data
│   ├── eda.ipynb               # exploratory data analysis and decisions
│   └── aqi_prediction.ipynb    # feature engineering, collinearity (VIF) check
├── experiments/                # model experiments, run in order (see "How to run")
│   ├── baseline_forecast.ipynb # persistence and other simple baselines
│   ├── lr_train_baseline.ipynb # logistic regression
│   ├── dt_train.ipynb          # decision tree
│   ├── rf_train.ipynb          # random forest
│   ├── xgb_train.ipynb         # XGBoost
│   ├── model_comparison.ipynb  # ROC AUC of every model
│   └── models/                 # models saved by the experiments, for comparison only
├── scripts/
│   ├── utility.py              # shared helpers used by the notebooks and train.py
│   ├── train.py                # trains the final model (XGBoost), tests it once on 2025
│   └── predict.py              # prediction service (to do)
├── models/                     # the final model only: xgb_final.bin, made by scripts/train.py
├── outputs/                    # every notebook run top to bottom, exported to Markdown
├── docs/
│   ├── EDA_Decision_Document.pdf          # every EDA decision, with evidence
│   ├── feature_transformation_notes.md    # reusable notes on encoding and transforms
│   ├── model_decisions.md                 # every modelling decision, with evidence
│   ├── model_comparison.md                # the four models compared, tuning settings
│   └── production_plan.md                 # how the forecast would run in production, on demand
├── problem_statement.pdf
└── requirements.txt
```

## Setup

Python 3.13 was used; any recent Python 3 should work.

```bash
git clone https://github.com/shohbitgupta/aqi_forcasting.git
cd aqi_forcasting

python3 -m venv .env
source .env/bin/activate            # Windows: .env\Scripts\activate
pip install -r requirements.txt

# Jupyter kernel used by the notebooks
pip install jupyter ipykernel
python -m ipykernel install --user --name aqi-forecasting --display-name "Python 3 (aqi_forcasting .env)"
```

## How to run

**Notebooks.** Start Jupyter (or open the folder in VS Code), choose the `aqi-forecasting` kernel and run the notebooks **in this order**.

From step 4 on, each experiment saves its tuned model in `experiments/models/`, and later experiments load the earlier models to compare with them. A notebook run before the ones it needs stops with a "file not found" error.

| # | Notebook | What it does | Saves in `experiments/models/` | Needs | Run time* |
|---|---|---|---|---|---|
| 1 | `notebooks/eda.ipynb` | Exploratory data analysis and decisions | — | — | seconds |
| 2 | `notebooks/aqi_prediction.ipynb` | Feature engineering, collinearity (VIF) check | — | — | seconds |
| 3 | `experiments/baseline_forecast.ipynb` | Persistence ("tomorrow = today") and other simple baselines | — | — | seconds |
| 4 | `experiments/lr_train_baseline.ipynb` | Logistic regression: class weights, tuning of `C` | `lr_baseline.bin`, `lr_no_weights.bin`, `lr_best.bin` | — | ~20 s |
| 5 | `experiments/dt_train.ipynb` | Decision tree: tuning, rolling validation 2019–2023 | `dt_best.bin` | `lr_best.bin` | ~1.5 min |
| 6 | `experiments/rf_train.ipynb` | Random forest: tuning, rolling validation | `rf_best.bin` (about 114 MB) | `lr_best.bin`, `dt_best.bin` | ~5 min |
| 7 | `experiments/xgb_train.ipynb` | XGBoost: tuning with early stopping, rolling validation | `xgb_best.bin` | `lr_best.bin`, `dt_best.bin`, `rf_best.bin` | ~8–11 min |
| 8 | `experiments/model_comparison.ipynb` | ROC AUC of every model, on 2023 and over five years | — | all of the above | ~1.5 min |

\* Measured on a 10-core laptop.

The notebooks read `../data/processed/aqi_weather_2015_2025.csv` and import the helpers from `../scripts/utility.py`, both relative to their own folder. The executed notebooks, with every output, are in `outputs/` as Markdown.

**Training the final model.** From the project root:

```bash
python scripts/train.py                          # -> models/xgb_final.bin
python scripts/train.py --output my_model.bin    # or choose where to save it
```

The script trains XGBoost with the settings chosen in `experiments/xgb_train.ipynb` on 2015–2024, tests it once on 2025 against persistence, prints the results and saves the model in `models/`. It takes about 10 seconds and does **not** need the experiments' model files.

**macOS: XGBoost needs OpenMP.** If `import xgboost` fails with `libomp.dylib could not be loaded`, install it with `brew install libomp`.

**Rebuilding the data (optional).** The data is already in the repository. To download it again, run the three steps from the project root, in order:

```bash
python code/prepare_aqi.py        # -> data/processed/aqi_daily.csv
python code/download_weather.py   # -> data/processed/weather_daily.csv (cached per city in data/raw/weather/)
python code/merge_data.py         # -> data/processed/aqi_weather_2015_2025.csv
```

`download_weather.py` calls the free Open-Meteo API (no key needed) one city-year at a time and caches each city, so it can be re-run safely if it stops.

## Approach

### Exploratory data analysis

The analysis is in [`notebooks/eda.ipynb`](notebooks/eda.ipynb), with every output visible in [`outputs/eda.md`](outputs/eda.md): value ranges and outliers, missing values, the target and its class balance, and how much each feature tells about tomorrow (box plots per class, Spearman correlation, mutual information, autocorrelation, collinearity). Every decision and its evidence is in [`docs/EDA_Decision_Document.pdf`](docs/EDA_Decision_Document.pdf). The main ones:

- **The target is built from the AQI number, never the text label.** 56 labels disagree with their own AQI value, 51 of them exactly on a category boundary.
- **Short gaps are filled, the target never is.** About 4% of days have no bulletin. Gaps of up to 3 days are filled with the last known value (older values are worse than a city + month average); longer gaps are dropped.
- **Classes are imbalanced.** Severe is 2.4% of days, so models are scored with **macro F1** and **recall on Very Poor and Severe**. Balanced class weights were tested for every model rather than assumed; the final XGBoost model uses none ([`docs/model_decisions.md`](docs/model_decisions.md), Decision 3).
- **Strong seasonality.** In November 10.4% of days are Severe; in July almost none.
- **Today's AQI is by far the strongest feature** (Spearman rho 0.88). Weather tells *whether* a bad day is coming but cannot tell Very Poor from Severe.
- **Outliers are kept.** They are real pollution episodes, found with a skew-aware rule instead of the plain 1.5 × IQR rule.
- **The split is by time,** so a model is always tested on days after the ones it learned from. Models are tuned on 2015–2022 with 2023 as validation, then checked on five separate years (2019–2023). The final model is trained on 2015–2024 and tested once on 2025.

### Feature engineering

Built in `notebooks/aqi_prediction.ipynb` (part 1) with `prepare_dataset` and `add_lr_features` from `scripts/utility.py`:

- **Lag and rolling AQI features per city**, using only today and earlier days: 7-day rolling mean and standard deviation, the change since yesterday, and today's deviation from the previous week.
- **Pollutant flags:** one 0/1 column per prominent pollutant.
- **For logistic regression only:**
  - month as `month_sin` / `month_cos`, so December and January are neighbours
  - `log1p` of rain, to shrink its long tail
  - pressure relative to each city's normal, which removes the altitude effect and raises its signal from rho 0.18 to 0.52
- **Collinearity check (VIF):** `aqi_today` with all four trend features is collinear (VIF ≈ 341), so logistic regression drops `aqi_roll7_mean`. Every remaining VIF is below 10.

Modelling dataset: 48,632 rows. Tuning: train 2015–2022 (33,543), validation 2023 (5,104). Final model: train 2015–2024 (43,755), test 2025 (4,877).

### Model training

Four models were trained, one linear and three tree-based, and each was tuned on 2023:

| Model | Notebook | Parameters tuned | Settings tried | Chosen |
|---|---|---|---|---|
| Logistic regression (linear) | [`lr_train_baseline.ipynb`](experiments/lr_train_baseline.ipynb) | `C` (0.0001–100) × class weights | 14 | `C = 1`, no class weights |
| Decision tree | [`dt_train.ipynb`](experiments/dt_train.ipynb) | `max_depth` × `min_samples_leaf` × class weights | 60 | depth 10, leaf 100, no class weights |
| Random forest | [`rf_train.ipynb`](experiments/rf_train.ipynb) | `max_features` × `min_samples_leaf` × class weights; `n_estimators` from a tree-count curve | 32 | 200 trees, `max_features` 0.5, leaf 5, `balanced_subsample` |
| XGBoost | [`xgb_train.ipynb`](experiments/xgb_train.ipynb) | `max_depth` × `min_child_weight` × class weights; `n_estimators` by early stopping (`learning_rate` fixed at 0.05) | 12 | depth 3, `min_child_weight` 10, 628 rounds, no class weights |

**One good year is not enough.** The best settings of each model were re-checked with **rolling validation**: learn from the years before, score one year, for each year from 2019 to 2023. A model counts as better only if it beats persistence on the five-year average **and** in most of the years.

| Five years, 2019–2023 | Persistence | Logistic regression | Decision tree | Random forest | XGBoost |
|---|---|---|---|---|---|
| Macro F1, average | 0.634 | 0.613 | 0.631 | **0.664** | 0.655 |
| Years ahead of persistence | — | 1 of 5 | 3 of 5 | **5 of 5** | **5 of 5** |
| Macro ROC AUC, average | — | 0.935 | 0.930 | 0.940 | **0.942** |

**Final model: XGBoost.** It is one of the two models that beat persistence in every year, it ranks risky days best (highest AUC in every year), and its file is 27 times smaller than the random forest's (4 MB against 114 MB). [`scripts/train.py`](scripts/train.py) trains it with these settings on 2015–2024 and tests it once on 2025.

Full comparison and every tuning grid: [`docs/model_comparison.md`](docs/model_comparison.md). The reasons behind each choice: [`docs/model_decisions.md`](docs/model_decisions.md). The executed notebooks: [`outputs/`](outputs/).

## Evaluation

- **Baseline to beat:** persistence, i.e. tomorrow's category = today's.
- **Metrics:** macro F1, confusion matrix, and recall on Very Poor and Severe.
- **Watch:** Very Poor and Severe days become rarer over time (Severe is 3.1% of training days but 1.0% of validation days), so Severe recall on validation rests on 51 days, and precision on those classes needs checking too.

## Production plan

In production the forecast is **on demand**: a user picks a city in the application and gets the forecast for the next bulletin, with the date it is for. At go-live the past bulletins are stored once; after that, a **daily refresh** fetches only the new bulletin and weather (about 4:30 PM IST, catching up any missed day) and prepares each city's model input. A user request reads that input and calls a **stateless prediction service**, so it takes milliseconds and does not depend on CPCB being online. Every day's forecasts are checked against the next bulletin. Phase 1, the prediction service, is the course deliverable; it can run on AWS Lambda within the free tier. Details: [`docs/production_plan.md`](docs/production_plan.md).

## Status

- [x] Data pipeline
- [x] Exploratory data analysis
- [x] Feature engineering
- [x] Persistence baseline and logistic regression baseline
- [x] Tree models (decision tree, random forest, XGBoost) and tuning
- [x] Final model chosen: XGBoost ([`docs/model_comparison.md`](docs/model_comparison.md))
- [x] `scripts/train.py` and the final evaluation on the 2025 test set
- [ ] `scripts/predict.py`
- [ ] Web service (FastAPI or Flask) in Docker, returning the predicted category and class probabilities
