<!-- Exported from experiments/accuracy_experiments.ipynb: every cell run top to bottom; generated, do not edit by hand -->

# Accuracy Experiments: Can We Forecast the Right Category More Often?

**The goal of this notebook is overall accuracy**: how often the forecast category is the one CPCB reports the next day. The final model (`scripts/train.py`) is right on 75% of 2025 days; "tomorrow = today" (persistence) on about 73%. In the live weeks of October 2026 the model was close to persistence outside the NCR.

**Two ideas are tested, plus two diagnostics:**

| # | Idea | Why it might help |
|---|---|---|
| 1 | **Recent years count more** (sample weights that halve every 3 or 6 years back) | air in several cities is cleaner than in 2015-2018, and the model over-forecast Moderate in October 2026 |
| 2 | **Forecast the AQI number**, then turn it into the category | the model would know that 291 is almost 301; most misses are near a category boundary |
| diag | **AQI 2 and 3 days ago** | a longer memory of the trend |
| diag | **Perfect tomorrow's weather** (the weather that really happened) | **cannot be deployed**: it only shows the most any weather forecast could add |

**The rules stay the same as for every model so far:**
- choose with **rolling validation on 2019-2023**: each year is forecast by a model that learned only from the years before it;
- the test year **2025 is scored once**, after the choice;
- always next to **persistence**, and with accuracy split into **days the category changed** and **steady days**;
- one more check no model has seen: **31 live days in October 2026**, built from the CPCB bulletins exactly as the web service does.

## 1. Setup


```python
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, '../scripts')                 # utility.py and train.py live in scripts/
import train as T                                # the final model's features, settings and data loading
from utility import AQI_CATEGORIES, CategoryXGB, aqi_to_category

warnings.filterwarnings('ignore', category=UserWarning)
DATA_PATH = Path('../data/processed/aqi_weather_2015_2025.csv')
LIVE_PATH = Path('../data/processed/live_check_2026_10.csv')
VAL_YEARS = range(2019, 2024)
NCR = ['Delhi', 'Noida', 'Greater Noida', 'Ghaziabad', 'Gurugram', 'Faridabad']
NORTH = ['Lucknow', 'Kanpur', 'Patna']
```

## 2. The data, and three extra columns

The same rows as every model (`train.load_data`). Three columns are added, only for this notebook:
- **`tomorrow_aqi`**: the next day's AQI **number**, the target of idea 2. `tomorrow_status` was made from it, so the two must agree on every row;
- **`aqi_lag2`, `aqi_lag3`**: the AQI 2 and 3 days earlier (diagnostic);
- **`next_*`**: the next day's measured weather (the ceiling diagnostic only).


```python
df = T.load_data(DATA_PATH)
raw = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values(['city', 'date'])
by_city = raw.groupby('city')['aqi_value']
raw['tomorrow_aqi'] = by_city.shift(-1)
raw['aqi_lag2'], raw['aqi_lag3'] = by_city.shift(2), by_city.shift(3)

WEATHER = ['temp_mean', 'temp_max', 'temp_min', 'humidity_mean', 'dew_point_mean', 'precip_sum', 'pressure_mean',
           'cloud_cover_mean', 'wind_speed_mean', 'wind_speed_max', 'wind_u_mean', 'wind_v_mean']
next_day = raw[['city', 'date'] + WEATHER].copy()
next_day['date'] -= pd.Timedelta(days=1)                 # the next day's weather, on today's row
next_day = next_day.rename(columns={w: f'next_{w}' for w in WEATHER})

df = (df.merge(raw[['city', 'date', 'tomorrow_aqi', 'aqi_lag2', 'aqi_lag3']], on=['city', 'date'], how='left')
        .merge(next_day, on=['city', 'date'], how='left'))
agree = (aqi_to_category(df['tomorrow_aqi']).astype(str) == df['tomorrow_status'].astype(str)).mean()
print(f'{len(df):,} rows; tomorrow_aqi agrees with tomorrow_status on {agree:.2%} of them')
```

    48,632 rows; tomorrow_aqi agrees with tomorrow_status on 100.00% of them


## 3. The candidates

Every candidate uses the final model's settings (`train.PARAMS`: depth 3, 628 trees, learning rate 0.05), so only the idea changes.

- **current**: the final model as it is.
- **recent_3 / recent_6**: the same model, with each training day weighted `0.5 ** (years back / half-life)`. With a half-life of 3 years, a day from 3 years before the last training year counts half as much.
- **aqi_number**: an XGBoost **regressor** forecasts tomorrow's AQI **change** (tomorrow minus today), the forecast number is today's AQI plus that change, and the category comes from CPCB's bands. Forecasting the change rather than the level lets the trees spend their splits on what moves.
- **aqi_number_3**: the same, with recent years weighted (half-life 3).
- **lags** and **perfect_weather** are the diagnostics.


```python
REG_PARAMS = {k: v for k, v in T.PARAMS.items() if k != 'balanced'}

def recency_weights(part, half_life):
    years = part['date'].dt.year.to_numpy()
    return 0.5 ** ((years.max() - years) / half_life)

def fit_predict(name, train, val):
    """Category forecasts for `val` from a candidate trained on `train`, and its AQI numbers (number models only)."""
    feats = (T.FEATURES + ['aqi_lag2', 'aqi_lag3'] if name == 'lags' else
             T.FEATURES + [f'next_{w}' for w in WEATHER] if name == 'perfect_weather' else T.FEATURES)
    X, X_val = train[feats].to_dict('records'), val[feats].to_dict('records')
    weights = recency_weights(train, int(name[-1])) if name[-1].isdigit() else None
    if name.startswith('aqi_number'):
        model = Pipeline([('dv', DictVectorizer(sparse=False)),
                          ('m', XGBRegressor(**REG_PARAMS, tree_method='hist', random_state=1, n_jobs=-1))])
        model.fit(X, (train['tomorrow_aqi'] - train['aqi_today']).to_numpy(), m__sample_weight=weights)
        aqi = np.clip(val['aqi_today'].to_numpy() + model.predict(X_val), 0, 500)
        return aqi_to_category(pd.Series(aqi)).astype(str).to_numpy(), aqi
    model = Pipeline([('dv', DictVectorizer(sparse=False)), ('m', CategoryXGB(**T.PARAMS))])
    model.fit(X, train[T.TARGET].astype(str).to_numpy(), m__sample_weight=weights)
    return model.predict(X_val), None

def scores(pred, part, aqi=None):
    """Accuracy overall, on days the category changed and on steady days; the AQI error for number models."""
    y = part[T.TARGET].astype(str).to_numpy()
    changed = part['status_today'].astype(str).to_numpy() != y
    out = {'accuracy': (pred == y).mean(), 'on change days': (pred[changed] == y[changed]).mean(),
           'on steady days': (pred[~changed] == y[~changed]).mean()}
    if aqi is not None:
        out['AQI error (MAE)'] = np.abs(aqi - part['tomorrow_aqi'].to_numpy()).mean()
    return out

CANDIDATES = ['current', 'recent_3', 'recent_6', 'aqi_number', 'aqi_number_3', 'lags', 'perfect_weather']
```

## 4. Rolling validation, 2019-2023

Each year is forecast by models that learned only from the years before it, so every score imitates real use. This takes a few minutes (7 candidates x 5 years).


```python
rows, val_forecasts = [], {}
for year in VAL_YEARS:
    train, val = df[df['date'].dt.year < year], df[df['date'].dt.year == year]
    rows.append({'year': year, 'model': 'persistence', **scores(val['status_today'].astype(str).to_numpy(), val)})
    for name in CANDIDATES:
        pred, aqi = fit_predict(name, train, val)
        rows.append({'year': year, 'model': name, **scores(pred, val, aqi)})
        if name == 'current':
            val_forecasts[year] = pred
rolling = pd.DataFrame(rows)
by_year = rolling.pivot(index='year', columns='model', values='accuracy')[['persistence'] + CANDIDATES]
by_year.round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>persistence</th>
      <th>current</th>
      <th>recent_3</th>
      <th>recent_6</th>
      <th>aqi_number</th>
      <th>aqi_number_3</th>
      <th>lags</th>
      <th>perfect_weather</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019</th>
      <td>0.629</td>
      <td>0.671</td>
      <td>0.671</td>
      <td>0.672</td>
      <td>0.662</td>
      <td>0.663</td>
      <td>0.670</td>
      <td>0.686</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>0.694</td>
      <td>0.722</td>
      <td>0.724</td>
      <td>0.726</td>
      <td>0.719</td>
      <td>0.722</td>
      <td>0.726</td>
      <td>0.729</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>0.679</td>
      <td>0.717</td>
      <td>0.716</td>
      <td>0.720</td>
      <td>0.711</td>
      <td>0.714</td>
      <td>0.718</td>
      <td>0.731</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>0.694</td>
      <td>0.722</td>
      <td>0.725</td>
      <td>0.721</td>
      <td>0.723</td>
      <td>0.717</td>
      <td>0.723</td>
      <td>0.729</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>0.710</td>
      <td>0.735</td>
      <td>0.737</td>
      <td>0.737</td>
      <td>0.737</td>
      <td>0.735</td>
      <td>0.735</td>
      <td>0.745</td>
    </tr>
  </tbody>
</table>
</div>




```python
summary = rolling.groupby('model', sort=False).mean(numeric_only=True).drop(columns='year')
summary['vs current'] = summary['accuracy'] - summary.loc['current', 'accuracy']
summary['years better than current'] = [int((by_year[m] > by_year['current']).sum()) for m in summary.index]
summary.round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>on change days</th>
      <th>on steady days</th>
      <th>AQI error (MAE)</th>
      <th>vs current</th>
      <th>years better than current</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>persistence</th>
      <td>0.681</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>NaN</td>
      <td>-0.032</td>
      <td>0</td>
    </tr>
    <tr>
      <th>current</th>
      <td>0.713</td>
      <td>0.304</td>
      <td>0.904</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>recent_3</th>
      <td>0.715</td>
      <td>0.314</td>
      <td>0.901</td>
      <td>NaN</td>
      <td>0.001</td>
      <td>4</td>
    </tr>
    <tr>
      <th>recent_6</th>
      <td>0.715</td>
      <td>0.312</td>
      <td>0.903</td>
      <td>NaN</td>
      <td>0.002</td>
      <td>4</td>
    </tr>
    <tr>
      <th>aqi_number</th>
      <td>0.710</td>
      <td>0.334</td>
      <td>0.886</td>
      <td>24.476</td>
      <td>-0.003</td>
      <td>2</td>
    </tr>
    <tr>
      <th>aqi_number_3</th>
      <td>0.710</td>
      <td>0.335</td>
      <td>0.885</td>
      <td>24.399</td>
      <td>-0.003</td>
      <td>0</td>
    </tr>
    <tr>
      <th>lags</th>
      <td>0.714</td>
      <td>0.306</td>
      <td>0.905</td>
      <td>NaN</td>
      <td>0.001</td>
      <td>3</td>
    </tr>
    <tr>
      <th>perfect_weather</th>
      <td>0.724</td>
      <td>0.343</td>
      <td>0.902</td>
      <td>NaN</td>
      <td>0.011</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



**Reading the table:**
- **Recency weights** add about **0.2 points**, better than the current model in 4 of 5 years, but by a margin too small to matter.
- **Forecasting the AQI number** catches **more changes** (about 33% instead of 30%) but makes **more mistakes on steady days**, and the two cancel out: overall it is slightly worse. Its typical AQI error is about 24 points.
- **AQI 2-3 days ago** adds nothing: the 7-day features already hold that memory.
- **Perfect tomorrow's weather** adds only about **1 point**, in every year. That is the **ceiling** for adding weather forecasts: real forecasts are less accurate than the weather that really happened, so they would add less.

So none of the deployable ideas moves overall accuracy by more than a fraction of a point. The next section shows why.

## 5. Why accuracy stops in the low 70s

A category is a band of AQI (Satisfactory is 51-100, Moderate 101-200, ...). A natural guess is that the misses are the days when tomorrow's AQI lands close to a band's edge. The next cell checks that guess against the current model's misses in 2019-2023.


```python
BOUNDS = np.array([50, 100, 200, 300, 400])
val_all = df[df['date'].dt.year.isin(VAL_YEARS)].copy()
val_all['forecast'] = np.concatenate([val_forecasts[y] for y in VAL_YEARS])
val_all['distance'] = val_all['tomorrow_aqi'].apply(lambda a: np.abs(BOUNDS - a).min())
miss = val_all[val_all['forecast'] != val_all[T.TARGET].astype(str)]
for d in (10, 20, 30):
    print(f'misses where tomorrow\'s AQI was within {d:>2} points of a category edge: {(miss["distance"] <= d).mean():.0%}'
          f'   (all days: {(val_all["distance"] <= d).mean():.0%})')
```

    misses where tomorrow's AQI was within 10 points of a category edge: 32%   (all days: 32%)
    misses where tomorrow's AQI was within 20 points of a category edge: 59%   (all days: 60%)
    misses where tomorrow's AQI was within 30 points of a category edge: 76%   (all days: 77%)


**The guess is wrong:** misses are no closer to an edge than days in general (32% of misses are within 10 points of one, and so are 32% of all days). The edges are not the special cause: the forecast is simply uncertain on many days.

A thought experiment shows how much certainty a category forecast needs: suppose a forecaster knew tomorrow's AQI **number** with a typical error of *e* points. How often would its category be right?


```python
rng = np.random.default_rng(1)
truth = val_all['tomorrow_aqi'].to_numpy()
true_cat = aqi_to_category(pd.Series(truth)).astype(str).to_numpy()
for e in (5, 10, 15, 20, 25):
    guess = np.clip(truth + rng.laplace(0, e, len(truth)), 0, 500)      # Laplace noise: mean absolute error = e
    acc = (aqi_to_category(pd.Series(guess)).astype(str).to_numpy() == true_cat).mean()
    print(f'AQI number known to within about {e:>2} points  ->  category right on {acc:.0%} of days')
```

    AQI number known to within about  5 points  ->  category right on 93% of days
    AQI number known to within about 10 points  ->  category right on 85% of days
    AQI number known to within about 15 points  ->  category right on 79% of days
    AQI number known to within about 20 points  ->  category right on 73% of days
    AQI number known to within about 25 points  ->  category right on 69% of days


Our number model is off by **about 24 points** on a typical day (section 4), and a forecaster that far off gets the category right on about **70%** of days, which is where every candidate lands. **To reach 85%, tomorrow's AQI would have to be known to within about 10 points**: far more than yesterday's AQI, the past week and the weather can tell. The limit is how much tomorrow's AQI moves for reasons these inputs do not see, not the choice of model.

## 6. The test year 2025, scored once

Each candidate is trained on 2015-2024 and scored on 2025, with persistence alongside, overall and by region.


```python
train, test = df[df['date'].dt.year <= T.TRAIN_END], df[df['date'].dt.year == T.TEST_YEAR]
region = np.select([test['city'].isin(NCR), test['city'].isin(NORTH)], ['NCR', 'Lucknow/Kanpur/Patna'], 'South & West')
test_rows, test_forecasts = [{'model': 'persistence', **scores(test['status_today'].astype(str).to_numpy(), test)}], {}
for name in ['current', 'recent_3', 'aqi_number']:
    pred, aqi = fit_predict(name, train, test)
    test_forecasts[name] = pred
    test_rows.append({'model': name, **scores(pred, test, aqi)})
pd.DataFrame(test_rows).set_index('model').round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>on change days</th>
      <th>on steady days</th>
      <th>AQI error (MAE)</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>persistence</th>
      <td>0.733</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>current</th>
      <td>0.751</td>
      <td>0.281</td>
      <td>0.922</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>recent_3</th>
      <td>0.758</td>
      <td>0.299</td>
      <td>0.925</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>aqi_number</th>
      <td>0.742</td>
      <td>0.326</td>
      <td>0.894</td>
      <td>20.337</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_test = test[T.TARGET].astype(str).to_numpy()
pd.DataFrame({name: pd.Series(pred == y_test).groupby(region).mean()
              for name, pred in [('persistence', test['status_today'].astype(str).to_numpy()), *test_forecasts.items()]}).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>persistence</th>
      <th>current</th>
      <th>recent_3</th>
      <th>aqi_number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lucknow/Kanpur/Patna</th>
      <td>0.750</td>
      <td>0.754</td>
      <td>0.758</td>
      <td>0.742</td>
    </tr>
    <tr>
      <th>NCR</th>
      <td>0.663</td>
      <td>0.692</td>
      <td>0.704</td>
      <td>0.691</td>
    </tr>
    <tr>
      <th>South &amp; West</th>
      <td>0.799</td>
      <td>0.813</td>
      <td>0.816</td>
      <td>0.797</td>
    </tr>
  </tbody>
</table>
</div>



## 7. Live October 2026: days no model has seen

The inputs for 3 October to 2 November 2026 were built from the CPCB bulletin PDFs and Open-Meteo exactly as the web service builds them (`scripts/live_data.py`), with the next day's CPCB AQI as the answer. If the file is missing, the next cell rebuilds it: about 40 bulletins, and one weather request per city for the whole month (the daily values are the same as the service's one-day requests, since the hourly data is averaged per date).

Here a candidate can also learn from **2025**, which is what a model retrained today would do.


```python
if not LIVE_PATH.exists():                       # rebuild from the bulletins, as the web service would
    import contextlib, io
    from datetime import date, timedelta
    import live_data
    month, one_day = {}, live_data.fetch_hourly
    def fetch_month(lat, lon, start, end, retries=2):            # one request per city, sliced per day
        if (lat, lon) not in month:
            month[(lat, lon)] = one_day(lat, lon, '2026-10-01', '2026-11-02', retries=5)
        h = month[(lat, lon)]
        return h[(h['time'] >= start) & (h['time'] < f'{end}T23:59')]
    live_data.fetch_hourly = fetch_month
    built = []
    for city in sorted(df['city'].unique()):
        for i in range(31):
            day = date(2026, 10, 3) + timedelta(days=i)
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    rec = live_data.build_record(city, day)
                except RuntimeError:
                    continue
                built.append({**rec, 'date': day, 'tomorrow_aqi': live_data.actual_aqi(city, day + timedelta(days=1))})
    pd.DataFrame(built).to_csv(LIVE_PATH, index=False)

live = pd.read_csv(LIVE_PATH, parse_dates=['date']).dropna(subset=['tomorrow_aqi'])
live[T.TARGET] = aqi_to_category(live['tomorrow_aqi']).astype(str)
live['status_today'] = aqi_to_category(live['aqi_today']).astype(str)
print(f'{len(live)} live city-days, {live["date"].min():%d %b} to {live["date"].max():%d %b %Y} (forecasts for the next day)')
```

    433 live city-days, 03 Oct to 02 Nov 2026 (forecasts for the next day)



```python
live_rows = [{'model': 'persistence', 'trained on': '-', **scores(live['status_today'].to_numpy(), live)}]
for years, label in [(T.TRAIN_END, '2015-2024'), (T.TEST_YEAR, '2015-2025')]:
    part = df[df['date'].dt.year <= years]
    for name in ['current', 'recent_3', 'aqi_number']:
        pred, aqi = fit_predict(name, part, live)
        live_rows.append({'model': name, 'trained on': label, **scores(pred, live, aqi)})
pd.DataFrame(live_rows).set_index(['model', 'trained on']).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>accuracy</th>
      <th>on change days</th>
      <th>on steady days</th>
      <th>AQI error (MAE)</th>
    </tr>
    <tr>
      <th>model</th>
      <th>trained on</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>persistence</th>
      <th>-</th>
      <td>0.691</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>current</th>
      <th>2015-2024</th>
      <td>0.714</td>
      <td>0.254</td>
      <td>0.920</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>recent_3</th>
      <th>2015-2024</th>
      <td>0.723</td>
      <td>0.276</td>
      <td>0.923</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>aqi_number</th>
      <th>2015-2024</th>
      <td>0.707</td>
      <td>0.306</td>
      <td>0.886</td>
      <td>23.315</td>
    </tr>
    <tr>
      <th>current</th>
      <th>2015-2025</th>
      <td>0.727</td>
      <td>0.269</td>
      <td>0.933</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>recent_3</th>
      <th>2015-2025</th>
      <td>0.730</td>
      <td>0.306</td>
      <td>0.920</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>aqi_number</th>
      <th>2015-2025</th>
      <td>0.709</td>
      <td>0.284</td>
      <td>0.900</td>
      <td>22.683</td>
    </tr>
  </tbody>
</table>
</div>



## 8. Conclusions


```python
print('Rolling validation 2019-2023, mean accuracy:')
print(summary[['accuracy', 'vs current', 'on change days', 'on steady days']].round(3).to_string())
```

    Rolling validation 2019-2023, mean accuracy:
                     accuracy  vs current  on change days  on steady days
    model                                                                
    persistence         0.681      -0.032           0.000           1.000
    current             0.713       0.000           0.304           0.904
    recent_3            0.715       0.001           0.314           0.901
    recent_6            0.715       0.002           0.312           0.903
    aqi_number          0.710      -0.003           0.334           0.886
    aqi_number_3        0.710      -0.003           0.335           0.885
    lags                0.714       0.001           0.306           0.905
    perfect_weather     0.724       0.011           0.343           0.902


**What was tested, and what moved overall accuracy:**

| Candidate | Validation 2019-2023 | Test 2025 | Live Oct 2026 |
|---|---|---|---|
| persistence ("tomorrow = today") | 68.1% | 73.3% | 69.1% |
| **current** final model | 71.3% | 75.1% | 71.4% |
| recent years weighted (half-life 3) | 71.5% | **75.8%** | 72.3% |
| AQI number, then category | 71.0% | 74.2% | 70.7% |
| AQI 2-3 days ago | 71.4% | - | - |
| perfect tomorrow's weather (not deployable) | 72.4% | - | - |
| current, **retrained with 2025** | - | - | 72.7% |
| recent years weighted, **retrained with 2025** | - | - | **73.0%** |

**Conclusions**
1. **Weighting recent years helps a little, every time it was measured**: +0.2 points in validation (better in 4 of 5 years), +0.7 on 2025 (NCR +1.2), +0.9 on the live days. Half-lives of 3 and 6 years are tied in validation; 3 was chosen before the test.
2. **Learning from 2025 too** adds about another point on the live days. Together with recency weights: **73.0% instead of 71.4%** on October 2026, and 3.9 points above persistence.
3. **Forecasting the AQI number does not raise category accuracy**: it catches more changes but misses more steady days. Its number (typical error 20-24 points) could still be shown in the app next to the category, as a separate decision.
4. **A longer AQI memory adds nothing**, and **even perfect knowledge of tomorrow's weather would add only about 1 point**, so adding weather forecasts is not worth the effort for overall accuracy.
5. **The ceiling is the data, not the model** (section 5): with these inputs tomorrow's AQI cannot be known to much better than about 20 points, which caps category accuracy in the low-to-mid 70s.

**Next step:** retrain the final model on **2015-2025 with recent years weighted (half-life 3)**. The expected gain is modest, about +0.5 to +1.5 points, and the change is small: `train.py` gains the weights and one more training year, and nothing changes in the service's inputs. With 2025 in training there is no unseen test year left, so the live checks (the web service's daily `check`) become the test.
