# Next-Day AQI Forecasting for Indian Cities

Predict **tomorrow's air-quality category** (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) for 14 Indian cities, using today's AQI, the pollutants driving it, the weather and the calendar.

DataTalks.Club ML Zoomcamp midterm project. Full problem statement: [`problem_statement.pdf`](problem_statement.pdf).

## Why it matters

Air-quality dashboards in India mostly report what the air is like *right now*. By the time the AQI reaches "Severe", people have already been exposed. A next-day forecast lets citizens plan outdoor activity, schools decide on closures, hospitals prepare, and authorities start pollution controls (Delhi's GRAP) **before** the peak rather than after.

The task is a **multi-class classification** problem. The target is the CPCB category of the next day's AQI:

| AQI | 0–50 | 51–100 | 101–200 | 201–300 | 301–400 | 401–500 |
|---|---|---|---|---|---|---|
| Category | Good | Satisfactory | Moderate | Poor | Very Poor | Severe |

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
├── notebooks/
│   ├── eda.ipynb               # exploratory data analysis and decisions
│   ├── aqi_prediction.ipynb    # feature engineering (part 1), modelling (part 2, in progress)
│   ├── utility.py              # shared helpers used by both notebooks
│   ├── train.py                # training script (to do)
│   └── predict.py              # prediction service (to do)
├── docs/
│   ├── EDA_Decision_Document.pdf          # every EDA decision, with evidence
│   └── feature_transformation_notes.md    # reusable notes on encoding and transforms
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

**Notebooks.** Start Jupyter (or open the folder in VS Code), choose the `aqi-forecasting` kernel and run the notebooks in order:

1. `notebooks/eda.ipynb`
2. `notebooks/aqi_prediction.ipynb`

The notebooks read `../data/processed/aqi_weather_2015_2025.csv`, relative to the `notebooks/` folder.

**Rebuilding the data (optional).** The data is already in the repository. To download it again, run the three steps from the project root, in order:

```bash
python code/prepare_aqi.py        # -> data/processed/aqi_daily.csv
python code/download_weather.py   # -> data/processed/weather_daily.csv (cached per city in data/raw/weather/)
python code/merge_data.py         # -> data/processed/aqi_weather_2015_2025.csv
```

`download_weather.py` calls the free Open-Meteo API (no key needed) one city-year at a time and caches each city, so it can be re-run safely if it stops.

## Approach

### Exploratory data analysis

Every decision and its evidence is in [`docs/EDA_Decision_Document.pdf`](docs/EDA_Decision_Document.pdf). The main ones:

- **The target is built from the AQI number, never the text label.** 56 labels disagree with their own AQI value, 51 of them exactly on a category boundary.
- **Short gaps are filled, the target never is.** About 4% of days have no bulletin. Gaps of up to 3 days are filled with the last known value (older values are worse than a city + month average); longer gaps are dropped.
- **Classes are imbalanced.** Severe is 2.4% of days, so models are scored with **macro F1** and **recall on Very Poor and Severe**, and trained with balanced class weights.
- **Strong seasonality.** In November 10.4% of days are Severe; in July almost none.
- **Today's AQI is by far the strongest feature** (Spearman rho 0.88). Weather tells *whether* a bad day is coming but cannot tell Very Poor from Severe.
- **Outliers are kept.** They are real pollution episodes, found with a skew-aware rule instead of the plain 1.5 × IQR rule.
- **The split is by time:** train 2015–2022, validation 2023, test 2024–2025, so the model is always tested on days after the ones it learned from.

### Feature engineering

Built in `notebooks/aqi_prediction.ipynb` (part 1) with `prepare_dataset` and `add_lr_features` from `notebooks/utility.py`:

- **Lag and rolling AQI features per city**, using only today and earlier days: 7-day rolling mean and standard deviation, the change since yesterday, and today's deviation from the previous week.
- **Pollutant flags:** one 0/1 column per prominent pollutant.
- **For logistic regression only:**
  - month as `month_sin` / `month_cos`, so December and January are neighbours
  - `log1p` of rain, to shrink its long tail
  - pressure relative to each city's normal, which removes the altitude effect and raises its signal from rho 0.18 to 0.52
- **Collinearity check (VIF):** `aqi_today` with all four trend features is collinear (VIF ≈ 341), so logistic regression drops `aqi_roll7_mean`. Every remaining VIF is below 10.

Modelling dataset: 48,632 rows (train 33,543, validation 5,104, test 9,985).

## Evaluation

- **Baseline to beat:** persistence, i.e. tomorrow's category = today's.
- **Metrics:** macro F1, confusion matrix, and recall on Very Poor and Severe.
- **Watch:** Very Poor and Severe days become rarer over time (Severe is 3.1% of training days but 1.0% of validation and 0.7% of test days), so Severe recall on validation rests on 51 days, and precision on those classes needs checking too.

## Status

- [x] Data pipeline
- [x] Exploratory data analysis
- [x] Feature engineering
- [ ] Persistence baseline and logistic regression baseline
- [ ] Tree models (random forest, XGBoost) and tuning
- [ ] Final evaluation on the 2024–2025 test set
- [ ] `train.py` and `predict.py`
- [ ] Web service (FastAPI or Flask) in Docker, returning the predicted category and class probabilities
