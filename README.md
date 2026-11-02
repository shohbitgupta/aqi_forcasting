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
| Recent daily AQI bulletins, for live forecasts only | [CPCB](https://cpcb.gov.in/) daily bulletin PDFs, downloaded when a live forecast runs | Public bulletins, not stored in the repository |

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
│   │   └── bulletins/          # CPCB bulletin PDFs for live forecasts: the latest 11 days only, not in git
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
│   ├── utility.py              # shared helpers used by the notebooks and the scripts
│   ├── train.py                # trains the final model (XGBoost), tests it once on 2025
│   ├── predict.py              # the prediction web service (FastAPI), and a command line for forecasts
│   ├── test_service.py         # sends demo.json to the running service and prints the answer
│   └── validate_live.py        # builds the live input from CPCB bulletins and Open-Meteo (used by predict.py)
├── models/                     # the final model only: xgb_final.bin, made by scripts/train.py
├── outputs/                    # every notebook run top to bottom, exported to Markdown
│   └── validation/             # the inputs built by live forecasts, one JSON file per city and day
├── docs/
│   ├── EDA_Decision_Document.pdf          # every EDA decision, with evidence
│   ├── feature_transformation_notes.md    # reusable notes on encoding and transforms
│   ├── model_decisions.md                 # every modelling decision, with evidence
│   ├── model_comparison.md                # the four models compared, tuning settings
│   └── production_plan.md                 # how the forecast would run in production, on demand
├── deploy/
│   └── deploy_lambda.sh        # deploys the service to AWS Lambda in one command (optional)
├── demo.json                   # an example input for predict.py --input
├── Dockerfile                  # image for the prediction service (local use and AWS Lambda)
├── Dockerfile.render           # the same image for Render, where the service is deployed
├── pyproject.toml, uv.lock     # dependencies, pinned to the versions the model was trained with
├── problem_statement.pdf
└── requirements.txt            # older, unpinned list; use uv instead (see Setup)
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13 (uv downloads it if needed).

```bash
git clone https://github.com/shohbitgupta/aqi_forcasting.git
cd aqi_forcasting

uv sync                             # creates .venv with every dependency, at the versions in uv.lock
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# Jupyter kernel used by the notebooks
python -m ipykernel install --user --name aqi-forecasting --display-name "Python 3 (aqi_forcasting .venv)"
uv run --with jupyter jupyter lab   # to open the notebooks outside VS Code
```

The model file is a pickle, so the libraries are pinned to the exact versions it was trained with; other versions may fail to load it. `uv sync --no-dev` installs only what forecasting from a file or from the project data needs. The notebooks, the data download scripts and live forecasts need the dev dependencies too, which `uv sync` installs by default.

The commands below assume `.venv` is active. Without activating it, put `uv run` in front of them, for example `uv run python scripts/train.py`.

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

**The prediction service.** The model runs as a FastAPI web service. Start it from the project root:

```bash
uvicorn predict:app --app-dir scripts --port 8080
```

It loads the model once at start-up and has two endpoints:

| Endpoint | What it does |
|---|---|
| `GET /health` | Checks the service is up and lists the 14 supported cities |
| `POST /predict` | Takes one city-day's 27 fields as JSON (see [`demo.json`](demo.json)) and returns the next day's category, the probability of every category, P(Severe), P(Dangerous) and the alert |

Test it in the browser at http://localhost:8080/docs (the request is pre-filled with `demo.json`), or from a second terminal:

```bash
python scripts/test_service.py           # sends demo.json and prints the answer
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d @demo.json
```

```json
{
  "city": "Noida",
  "forecast": "Poor",
  "p_severe": 0.0015,
  "p_dangerous": 0.1081,
  "probabilities": {"Good": 0.0001, "Satisfactory": 0.0072, "Moderate": 0.1741,
                    "Poor": 0.7105, "Very Poor": 0.1065, "Severe": 0.0015},
  "alert": false,
  "alert_rule": "forecast is below Very Poor"
}
```

- **The alert:** by default, `alert` is true when the forecast is Very Poor or Severe. Each user can set an earlier warning level instead, for example `POST /predict?dangerous_threshold=0.1` raises it when P(Very Poor or Severe) is at least 10%. `severe_threshold` does the same for P(Severe).
- **Bad input** (a missing field, a wrong type, `month` 13, an unknown city) gets an HTTP 422 answer that says what is wrong.

**Running the service in Docker.** The image holds the service, the model and the pinned libraries (installed from `uv.lock`), and nothing else: no data or notebooks. From the project root, with Docker running:

```bash
docker build -t aqi-forecast .
docker run -p 8080:8080 aqi-forecast
python scripts/test_service.py           # in a second terminal, the same test as above
```

There are two Dockerfiles; both build the same service and run locally the same way:

| File | For | Difference |
|---|---|---|
| `Dockerfile` | local use and AWS Lambda | includes the AWS Lambda Web Adapter, which does nothing outside Lambda |
| `Dockerfile.render` | Render | no adapter; listens on the port Render gives it (`$PORT`), 8080 otherwise |

**The deployed service.** The service runs on [Render](https://render.com)'s free plan, built from `Dockerfile.render`:

**https://aqi-forcasting.onrender.com** (the interactive docs are at [/docs](https://aqi-forcasting.onrender.com/docs))

```bash
python scripts/test_service.py --url https://aqi-forcasting.onrender.com
```

The free plan puts the service to sleep after 15 minutes without requests, so the first request after a quiet spell can take up to a minute; the ones after it are instant.

To deploy your own copy: on Render, **New → Web Service**, connect the GitHub repository and choose **Language: Docker**, **Instance type: Free**, and under **Advanced** set **Dockerfile Path** to `./Dockerfile.render` and **Health Check Path** to `/health`. Every push to `main` deploys again.

`deploy/deploy_lambda.sh` deploys the same service to AWS Lambda instead, with a public Function URL, in one command (`./deploy/deploy_lambda.sh`). It needs the AWS CLI and an AWS account that allows ECR and Lambda; it was not used for the deployment above, because the accounts available blocked those services.

**Making a forecast from the command line.** `scripts/predict.py` also forecasts the next day's category for a city. The input can come from three places:

```bash
# offline: a day from the project's data (2015-2025), shown next to what actually happened
python scripts/predict.py --mode offline --city Delhi --date 2025-11-15

# live: the latest CPCB bulletins and weather (needs internet)
python scripts/predict.py --mode live --city Delhi                     # based on yesterday
python scripts/predict.py --mode live --city Noida --date 2026-10-27   # a past day, with what happened

# your own input: one city-day or a list, with the model's 27 fields (see demo.json)
python scripts/predict.py --input demo.json
```

```
Delhi, based on 2025-11-15: AQI 386 (Very Poor)
Forecast for 2025-11-16: Very Poor
Actual on 2025-11-16:   Very Poor, forecast correct
```

- Leave out `--city` to pick one from the list of 14. `--date` is the last day of data and the forecast is for the day after. It defaults to the city's last day in the data (offline) or to yesterday, the last complete day (live), so a live forecast is for today.
- **Offline:** only days in 2025, the test year, are a fair check; the model learned from 2015–2024, and the script says so when you pick one of those days.
- **Live:** the input is built exactly as for training, from the same sources:
  - the AQI of the last 8 days, read from the CPCB daily bulletin PDFs, the source the training data was compiled from;
  - 3 more days before them, so a missing day can be filled from the day before, as in training;
  - that day's weather from the same Open-Meteo archive.

  On 2025 days this gives the same input as the training data, field for field, including days filled from the day before. The input is saved in `outputs/validation/`, so `--input` can run it again, and only the bulletins of the latest window are kept in `data/raw/bulletins/`. Live mode needs the dev dependencies (`requests`, `pdfplumber`).
- **Why the CPCB bulletins:** an easier source was tried first, the Open-Meteo air-quality API, which gives pollution levels estimated by the CAMS model rather than measured. Its AQI was in the same category as CPCB's on only 29% of 2025 days, and forecasts made from it were right on 33% of days instead of 75%. The model needs the same kind of data it learned from.

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
- [x] `scripts/predict.py`: forecasts from a JSON input, the project data (offline) or live CPCB bulletins
- [x] Web service (FastAPI), returning the predicted category, class probabilities and alert
- [x] The service in Docker
- [x] Cloud deployment: https://aqi-forcasting.onrender.com
