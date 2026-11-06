<!-- Exported from notebooks/aqi_prediction.ipynb: every cell run top to bottom; generated, do not edit by hand -->

# Next-Day AQI Forecasting: Feature Engineering and Modelling

**Goal:** predict tomorrow's AQI category (`tomorrow_status`, six classes from *Good* to *Severe*) for 14 Indian cities.

Builds on `eda.ipynb` (decisions D1–D12 in `docs/EDA_Decision_Document.pdf`). Part 1 builds and checks the model features; part 2 trains the baselines.

**Part 1, feature engineering**
1. Setup
2. Build the dataset in leakage-safe order
3. Handle rows with empty trend features; split by time
4. Transforms for logistic regression
5. Rerun the VIF check with the lag and rolling features
6. Final feature sets

## 1. Setup

```python
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../scripts')                 # utility.py lives in scripts/
from utility import (AQI_CATEGORIES, prepare_dataset, split_train_val_test,
                     add_lr_features, vif_table, class_signal_summary)

pd.set_option('display.max_columns', 40)
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True})
COLOR = '#2a78d6'

DATA_PATH = '../data/processed/aqi_weather_2015_2025.csv'
TARGET = 'tomorrow_status'
TREND = ['aqi_roll7_mean', 'aqi_roll7_std', 'aqi_diff1', 'aqi_dev7']
```

## 2. Build the dataset

`prepare_dataset` (in `utility.py`) applies the EDA decisions in a fixed order; steps 3 and 5 prevent leakage only in this sequence.

| Step | What | Why this position |
|---|---|---|
| 1 | Rename to `aqi_today` / `status_today` | Makes the time direction explicit |
| 2 | Pollutant flags `has_*` (NH<sub>3</sub> dropped) | D5 |
| 3 | `tomorrow_status` from the next day's AQI **number**, per city | **Before filling**, so a filled value never becomes a label (D1, D2) |
| 4 | `ffill_aqi(limit=3)`, adds `aqi_imputed` | Fills inputs only (D3) |
| 5 | `add_trend_features`: lag / rolling features + `month` | **Before dropping rows**, so a dropped day still counts as "yesterday" (D11) |
| 6 | Drop rows with no AQI today or no real tomorrow; drop the station count | D2, D3 |

The function reproduces the EDA notebook's data exactly (48,827 rows), plus the five new columns.

```python
df = prepare_dataset(DATA_PATH)
FLAGS = [c for c in df.columns if c.startswith('has_')]

WEATHER = ['temp_mean', 'temp_max', 'temp_min', 'temp_range', 'humidity_mean', 'dew_point_mean', 'precip_sum',
           'pressure_mean', 'cloud_cover_mean', 'wind_speed_mean', 'wind_speed_max', 'wind_u_mean', 'wind_v_mean',
           'wind_dir_mean']
groups = {'id': ['city', 'date'], 'AQI today': ['aqi_today', 'status_today', 'aqi_imputed'],
          'pollutant flags (D5)': FLAGS, 'weather': WEATHER, 'trend + calendar (D11)': TREND + ['month'],
          'target': [TARGET]}
assert sorted(sum(groups.values(), [])) == sorted(df.columns), 'every column belongs to one group'

print(f'Rows: {len(df):,} | columns: {df.shape[1]}')
pd.DataFrame({'columns': [', '.join(v) for v in groups.values()], 'count': [len(v) for v in groups.values()]},
             index=list(groups))
```

**Output:**

```
Rows: 48,827 | columns: 31
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>columns</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>city, date</td>
      <td>2</td>
    </tr>
    <tr>
      <th>AQI today</th>
      <td>aqi_today, status_today, aqi_imputed</td>
      <td>3</td>
    </tr>
    <tr>
      <th>pollutant flags (D5)</th>
      <td>has_co, has_no2, has_o3, has_pm10, has_pm25, h...</td>
      <td>6</td>
    </tr>
    <tr>
      <th>weather</th>
      <td>temp_mean, temp_max, temp_min, temp_range, hum...</td>
      <td>14</td>
    </tr>
    <tr>
      <th>trend + calendar (D11)</th>
      <td>aqi_roll7_mean, aqi_roll7_std, aqi_diff1, aqi_...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>target</th>
      <td>tomorrow_status</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
# A few rows across the feature groups
df[['city', 'date', 'aqi_today'] + FLAGS + ['aqi_imputed', 'month'] + TREND + [TARGET]].head(10)
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>date</th>
      <th>aqi_today</th>
      <th>has_co</th>
      <th>has_no2</th>
      <th>has_o3</th>
      <th>has_pm10</th>
      <th>has_pm25</th>
      <th>has_so2</th>
      <th>aqi_imputed</th>
      <th>month</th>
      <th>aqi_roll7_mean</th>
      <th>aqi_roll7_std</th>
      <th>aqi_diff1</th>
      <th>aqi_dev7</th>
      <th>tomorrow_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bengaluru</td>
      <td>2015-05-02</td>
      <td>132.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bengaluru</td>
      <td>2015-05-03</td>
      <td>49.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-83.0</td>
      <td>NaN</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bengaluru</td>
      <td>2015-05-04</td>
      <td>46.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-3.0</td>
      <td>NaN</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bengaluru</td>
      <td>2015-05-05</td>
      <td>229.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>114.000000</td>
      <td>86.406018</td>
      <td>183.0</td>
      <td>NaN</td>
      <td>Satisfactory</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengaluru</td>
      <td>2015-05-06</td>
      <td>90.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>109.200000</td>
      <td>75.595635</td>
      <td>-139.0</td>
      <td>-24.000000</td>
      <td>Satisfactory</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bengaluru</td>
      <td>2015-05-07</td>
      <td>52.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>99.666667</td>
      <td>71.533675</td>
      <td>-38.0</td>
      <td>-57.200000</td>
      <td>Satisfactory</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Bengaluru</td>
      <td>2015-05-08</td>
      <td>83.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>97.285714</td>
      <td>65.604152</td>
      <td>31.0</td>
      <td>-16.666667</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bengaluru</td>
      <td>2015-05-09</td>
      <td>46.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>85.000000</td>
      <td>66.070669</td>
      <td>-37.0</td>
      <td>-51.285714</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bengaluru</td>
      <td>2015-05-10</td>
      <td>266.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>116.000000</td>
      <td>92.132151</td>
      <td>220.0</td>
      <td>181.000000</td>
      <td>Satisfactory</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bengaluru</td>
      <td>2015-05-11</td>
      <td>98.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>123.428571</td>
      <td>87.528771</td>
      <td>-168.0</td>
      <td>-18.000000</td>
      <td>Satisfactory</td>
    </tr>
  </tbody>
</table>
</div>

The first rows of each city have empty trend features (a 7-day window needs at least 4 known days; the first day has no "yesterday"). Handled in section 3.

## 3. Rows with empty trend features

```python
empty = df[TREND].isna().any(axis=1)
print(f'Rows with at least one empty trend feature: {empty.sum()} ({empty.mean():.2%})')
print('By year:', df.loc[empty, 'date'].dt.year.value_counts().sort_index().to_dict())
pd.DataFrame({'all rows': df[TARGET].value_counts(normalize=True),
              'empty-trend rows': df.loc[empty, TARGET].value_counts(normalize=True)}).reindex(AQI_CATEGORIES).mul(100).round(1)
```

**Output:**

```
Rows with at least one empty trend feature: 195 (0.40%)
By year: {2015: 44, 2016: 66, 2017: 36, 2018: 35, 2019: 7, 2025: 7}
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>all rows</th>
      <th>empty-trend rows</th>
    </tr>
    <tr>
      <th>tomorrow_status</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Good</th>
      <td>8.3</td>
      <td>8.7</td>
    </tr>
    <tr>
      <th>Satisfactory</th>
      <td>32.8</td>
      <td>32.8</td>
    </tr>
    <tr>
      <th>Moderate</th>
      <td>31.0</td>
      <td>26.2</td>
    </tr>
    <tr>
      <th>Poor</th>
      <td>16.1</td>
      <td>17.9</td>
    </tr>
    <tr>
      <th>Very Poor</th>
      <td>9.4</td>
      <td>9.7</td>
    </tr>
    <tr>
      <th>Severe</th>
      <td>2.4</td>
      <td>4.6</td>
    </tr>
  </tbody>
</table>
</div>

**Finding:** 195 rows (0.4%) have an empty trend feature, 181 of them in 2015–2018 (each city's first days and the days after long gaps). Their class mix is close to the overall one, with slightly more Severe days (4.6% against 2.4%), but that is only about 9 of 1,178 Severe days, too few to change the balance.

**Decision:** drop them from every split. Logistic regression cannot take empty values, and a single dataset means every model is scored on the same days. (Tree models could handle the gaps, but 0.4% of rows does not justify two evaluation sets.)

```python
df = df[~empty].reset_index(drop=True)

df_train, df_val, df_test = split_train_val_test(df)
pd.DataFrame({'rows': [len(df_train), len(df_val), len(df_test)],
              'years': ['2015–2022', '2023', '2024–2025']}, index=['train', 'val', 'test'])
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rows</th>
      <th>years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>33543</td>
      <td>2015–2022</td>
    </tr>
    <tr>
      <th>val</th>
      <td>5104</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>test</th>
      <td>9985</td>
      <td>2024–2025</td>
    </tr>
  </tbody>
</table>
</div>

## 4. Transforms for logistic regression

Logistic regression needs roughly monotonic, linear effects; tree models split on thresholds and need none of this. Three EDA decisions become new columns (`add_lr_features`):

| New column | Problem fixed | EDA decision |
|---|---|---|
| `pressure_rel` = pressure − the city's normal pressure | Raw pressure mostly encodes the **city** (altitude): Bengaluru's normal is far below Delhi's | D8 |
| `precip_log` = log(1 + rain) | Rain is extremely skewed: 0 mm on most days, up to 251 mm; a few extreme values would dominate the coefficient | D8 |
| `month_sin`, `month_cos` | As an integer (1–12), December and January are far apart; cyclic encoding makes them adjacent | D7 |

The city's normal pressure is computed on the **training years only**, so nothing from 2023–2025 leaks into the features.

Rationale and reusable rules: [`docs/feature_transformation_notes.md`](../docs/feature_transformation_notes.md).

```python
city_pressure = df_train.groupby('city')['pressure_mean'].mean()     # training years only

df_train = add_lr_features(df_train, city_pressure)
df_val = add_lr_features(df_val, city_pressure)
df_test = add_lr_features(df_test, city_pressure)

new_vs_old = ['pressure_mean', 'pressure_rel', 'precip_sum', 'precip_log', 'month', 'month_cos', 'month_sin']
class_signal_summary(df_train, new_vs_old)[['rho', 'direction', 'medians', 'shape']]
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rho</th>
      <th>direction</th>
      <th>medians</th>
      <th>shape</th>
    </tr>
    <tr>
      <th>feature</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precip_sum</th>
      <td>-0.60</td>
      <td>down</td>
      <td>5.2 → 1.2 → 0 → 0 → 0 → 0</td>
      <td>bends</td>
    </tr>
    <tr>
      <th>precip_log</th>
      <td>-0.60</td>
      <td>down</td>
      <td>1.8 → 0.8 → 0 → 0 → 0 → 0</td>
      <td>bends</td>
    </tr>
    <tr>
      <th>pressure_rel</th>
      <td>0.52</td>
      <td>up</td>
      <td>-3.4 → -2.4 → 0.1 → 3.5 → 6.5 → 6.9</td>
      <td>straight</td>
    </tr>
    <tr>
      <th>month_cos</th>
      <td>0.48</td>
      <td>up</td>
      <td>-0.5 → -0.5 → 0 → 0.5 → 0.9 → 0.9</td>
      <td>bends</td>
    </tr>
    <tr>
      <th>month_sin</th>
      <td>0.21</td>
      <td>weak up</td>
      <td>-0.5 → -0.5 → 0 → 0.5 → -0 → -0</td>
      <td>reverses</td>
    </tr>
    <tr>
      <th>pressure_mean</th>
      <td>0.18</td>
      <td>weak up</td>
      <td>984.9 → 985.4 → 985.9 → 990.3 → 992.1 → 992</td>
      <td>bends</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.00</td>
      <td>none</td>
      <td>8 → 7 → 6 → 5 → 10 → 11</td>
      <td>reverses</td>
    </tr>
  </tbody>
</table>
</div>

```python
print('Skew of rain:  raw', round(df_train['precip_sum'].skew(), 2), '-> log', round(df_train['precip_log'].skew(), 2))
print('Raw month as a number, rho with the class order:',
      round(df_train['month'].rank().corr(df_train[TARGET].cat.codes.rank()), 2))
```

**Output:**

```
Skew of rain:  raw 7.8 -> log 1.57
Raw month as a number, rho with the class order: 0.0
```

**Findings:**
- **`pressure_rel` exposes a signal that raw pressure hid.** Raw pressure: rho 0.18. Relative to the city's normal: **rho 0.52**, linear. With altitude removed, the weather effect appears: higher-than-usual pressure → stagnant air → pollution builds up.
- **`month` as an integer has no linear signal (rho 0.00)**: air is bad in December *and* January, the two ends of the scale. `month_cos` (high in winter, low in the monsoon) reaches **rho 0.48**. `month_sin` is weak alone but separates autumn (October–November, crop burning) from spring, which share the same `month_cos`.
- **`precip_log`:** rho is unchanged (rank correlation is unaffected by a log), but skew drops from 7.8 to 1.6, so extreme rain days no longer dominate the coefficient.

**Decision:** logistic regression uses `pressure_rel`, `precip_log`, `month_sin` and `month_cos` instead of `pressure_mean`, `precip_sum` and `month`. Tree models keep the raw columns.

## 5. VIF with the lag and rolling features

The EDA's reduced set (D10) had every VIF below 10; the trend features are now added. Thresholds: about 1 = unique, 5–10 = moderate, above 10 = largely redundant. Training years only.

### 5.1 All four trend features together

```python
reduced_weather = ['temp_min', 'temp_range', 'humidity_mean', 'precip_sum', 'pressure_mean',
                   'cloud_cover_mean', 'wind_speed_mean', 'wind_u_mean', 'wind_v_mean']

vif_table(df_train[['aqi_today'] + TREND + reduced_weather]).to_frame('VIF: aqi_today + all trend features')
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF: aqi_today + all trend features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aqi_today</th>
      <td>340.64</td>
    </tr>
    <tr>
      <th>aqi_roll7_mean</th>
      <td>286.38</td>
    </tr>
    <tr>
      <th>aqi_dev7</th>
      <td>53.55</td>
    </tr>
    <tr>
      <th>temp_range</th>
      <td>7.43</td>
    </tr>
    <tr>
      <th>humidity_mean</th>
      <td>5.38</td>
    </tr>
    <tr>
      <th>temp_min</th>
      <td>3.68</td>
    </tr>
    <tr>
      <th>cloud_cover_mean</th>
      <td>2.37</td>
    </tr>
    <tr>
      <th>aqi_diff1</th>
      <td>1.70</td>
    </tr>
    <tr>
      <th>aqi_roll7_std</th>
      <td>1.52</td>
    </tr>
    <tr>
      <th>wind_speed_mean</th>
      <td>1.44</td>
    </tr>
    <tr>
      <th>pressure_mean</th>
      <td>1.39</td>
    </tr>
    <tr>
      <th>wind_v_mean</th>
      <td>1.36</td>
    </tr>
    <tr>
      <th>precip_sum</th>
      <td>1.31</td>
    </tr>
    <tr>
      <th>wind_u_mean</th>
      <td>1.09</td>
    </tr>
  </tbody>
</table>
</div>

**Finding:** three features explode: `aqi_today` (≈ 341), `aqi_roll7_mean` (≈ 286) and `aqi_dev7` (≈ 54). They are built from the same days:

- `aqi_roll7_mean` = the average of **today and the 6 days before**
- `aqi_dev7` = **today** minus the average of **the 7 days before**

Any two almost exactly determine the third, as with `temp_max`, `temp_min` and `temp_range`. One of the three has to go.

### 5.2 Which one to drop?

```python
option_a = ['aqi_today', 'aqi_roll7_std', 'aqi_diff1', 'aqi_dev7']        # drop aqi_roll7_mean
option_b = ['aqi_roll7_mean', 'aqi_roll7_std', 'aqi_diff1', 'aqi_dev7']   # drop aqi_today

pd.concat({'A: drop aqi_roll7_mean': vif_table(df_train[option_a + reduced_weather]),
           'B: drop aqi_today': vif_table(df_train[option_b + reduced_weather])}, axis=1)
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A: drop aqi_roll7_mean</th>
      <th>B: drop aqi_today</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>temp_range</th>
      <td>7.43</td>
      <td>7.43</td>
    </tr>
    <tr>
      <th>humidity_mean</th>
      <td>5.38</td>
      <td>5.38</td>
    </tr>
    <tr>
      <th>temp_min</th>
      <td>3.66</td>
      <td>3.64</td>
    </tr>
    <tr>
      <th>aqi_today</th>
      <td>3.55</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>cloud_cover_mean</th>
      <td>2.37</td>
      <td>2.37</td>
    </tr>
    <tr>
      <th>aqi_dev7</th>
      <td>2.13</td>
      <td>1.62</td>
    </tr>
    <tr>
      <th>aqi_diff1</th>
      <td>1.56</td>
      <td>1.57</td>
    </tr>
    <tr>
      <th>aqi_roll7_std</th>
      <td>1.52</td>
      <td>1.51</td>
    </tr>
    <tr>
      <th>wind_speed_mean</th>
      <td>1.44</td>
      <td>1.44</td>
    </tr>
    <tr>
      <th>pressure_mean</th>
      <td>1.38</td>
      <td>1.38</td>
    </tr>
    <tr>
      <th>wind_v_mean</th>
      <td>1.36</td>
      <td>1.36</td>
    </tr>
    <tr>
      <th>precip_sum</th>
      <td>1.31</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>wind_u_mean</th>
      <td>1.09</td>
      <td>1.09</td>
    </tr>
    <tr>
      <th>aqi_roll7_mean</th>
      <td>NaN</td>
      <td>2.99</td>
    </tr>
  </tbody>
</table>
</div>

```python
df_train[['aqi_today'] + TREND].corr().round(2)
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aqi_today</th>
      <th>aqi_roll7_mean</th>
      <th>aqi_roll7_std</th>
      <th>aqi_diff1</th>
      <th>aqi_dev7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aqi_today</th>
      <td>1.00</td>
      <td>0.91</td>
      <td>0.51</td>
      <td>0.21</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>aqi_roll7_mean</th>
      <td>0.91</td>
      <td>1.00</td>
      <td>0.57</td>
      <td>-0.05</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <th>aqi_roll7_std</th>
      <td>0.51</td>
      <td>0.57</td>
      <td>1.00</td>
      <td>-0.01</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>aqi_diff1</th>
      <td>0.21</td>
      <td>-0.05</td>
      <td>-0.01</td>
      <td>1.00</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>aqi_dev7</th>
      <td>0.38</td>
      <td>-0.03</td>
      <td>-0.05</td>
      <td>0.60</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>

**Finding:** both options bring every VIF back below 10, and the AQI features themselves to about 3.5 or lower.

**Decision: option A, drop `aqi_roll7_mean`.**
- `aqi_today` is the strongest single feature (rho 0.88 against 0.85) and the most recent information.
- `aqi_roll7_mean` correlates 0.91 with `aqi_today`: it is mostly a smoothed copy of it.
- What the weekly mean adds beyond today is kept in other forms: `aqi_dev7` (how far today is from last week) and `aqi_roll7_std` (how jumpy the week was).
- Tree models keep all four trend features.

### 5.3 The full candidate set for logistic regression

The final check combines option A with the transforms from section 4.

```python
LR_NUMERIC = ['aqi_today', 'aqi_roll7_std', 'aqi_diff1', 'aqi_dev7',
              'temp_min', 'temp_range', 'humidity_mean', 'precip_log', 'pressure_rel',
              'cloud_cover_mean', 'wind_speed_mean', 'wind_u_mean', 'wind_v_mean',
              'month_sin', 'month_cos']
LR_NUMERIC_NO_HUMIDITY = [c for c in LR_NUMERIC if c != 'humidity_mean']

pd.concat({'candidate set': vif_table(df_train[LR_NUMERIC]),
           'without humidity_mean': vif_table(df_train[LR_NUMERIC_NO_HUMIDITY])}, axis=1)
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>candidate set</th>
      <th>without humidity_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>temp_range</th>
      <td>8.07</td>
      <td>3.55</td>
    </tr>
    <tr>
      <th>temp_min</th>
      <td>7.50</td>
      <td>5.03</td>
    </tr>
    <tr>
      <th>humidity_mean</th>
      <td>5.89</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>pressure_rel</th>
      <td>5.51</td>
      <td>5.50</td>
    </tr>
    <tr>
      <th>month_cos</th>
      <td>4.89</td>
      <td>4.88</td>
    </tr>
    <tr>
      <th>aqi_today</th>
      <td>3.26</td>
      <td>3.20</td>
    </tr>
    <tr>
      <th>cloud_cover_mean</th>
      <td>2.52</td>
      <td>2.51</td>
    </tr>
    <tr>
      <th>precip_log</th>
      <td>2.39</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>month_sin</th>
      <td>2.16</td>
      <td>2.06</td>
    </tr>
    <tr>
      <th>aqi_dev7</th>
      <td>2.10</td>
      <td>2.07</td>
    </tr>
    <tr>
      <th>aqi_diff1</th>
      <td>1.57</td>
      <td>1.57</td>
    </tr>
    <tr>
      <th>aqi_roll7_std</th>
      <td>1.53</td>
      <td>1.53</td>
    </tr>
    <tr>
      <th>wind_speed_mean</th>
      <td>1.49</td>
      <td>1.32</td>
    </tr>
    <tr>
      <th>wind_v_mean</th>
      <td>1.44</td>
      <td>1.34</td>
    </tr>
    <tr>
      <th>wind_u_mean</th>
      <td>1.26</td>
      <td>1.24</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Why do temp_min, pressure_rel and month_cos rise? They describe the same thing: winter.
df_train[['temp_min', 'pressure_rel', 'month_cos', 'temp_range', 'humidity_mean']].corr().round(2)
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temp_min</th>
      <th>pressure_rel</th>
      <th>month_cos</th>
      <th>temp_range</th>
      <th>humidity_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>temp_min</th>
      <td>1.00</td>
      <td>-0.83</td>
      <td>-0.75</td>
      <td>-0.49</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>pressure_rel</th>
      <td>-0.83</td>
      <td>1.00</td>
      <td>0.81</td>
      <td>0.39</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>month_cos</th>
      <td>-0.75</td>
      <td>0.81</td>
      <td>1.00</td>
      <td>0.24</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>temp_range</th>
      <td>-0.49</td>
      <td>0.39</td>
      <td>0.24</td>
      <td>1.00</td>
      <td>-0.75</td>
    </tr>
    <tr>
      <th>humidity_mean</th>
      <td>-0.02</td>
      <td>-0.01</td>
      <td>0.14</td>
      <td>-0.75</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>

**Findings:**
- **Every VIF is below 10** in the candidate set. The highest are `temp_range` (≈ 8.1) and `temp_min` (≈ 7.5).
- **A "winter" group has formed:** cold nights (`temp_min`), higher-than-usual pressure (`pressure_rel`) and the winter months (`month_cos`) correlate at about ±0.8 with each other. Each still carries some information of its own, so all three stay, but their individual weights should be read with care.
- **Dropping `humidity_mean`** brings every VIF down to about 5.5 or lower (`temp_range` falls from 8.1 to 3.6), because humidity and `temp_range` move together (r = −0.75).
- The trend features stay low (1.5–3.3): they add information the weather does not have.

**Decisions:**
- **Baseline logistic regression:** `LR_NUMERIC` (15 numeric features), standardised, plus the categorical features below.
- **Variant to test on validation:** `LR_NUMERIC_NO_HUMIDITY`. Keep humidity only if the validation score drops without it.
- If coefficients flip sign between runs, the winter group is the first place to look.

## 6. Final feature sets

```python
LR_CATEGORICAL = ['city']                       # one-hot encoded in the model pipeline (D4)
LR_BINARY = FLAGS + ['aqi_imputed']

TREE_FEATURES = (['aqi_today'] + TREND +
                 ['temp_mean', 'temp_max', 'temp_min', 'temp_range', 'humidity_mean', 'dew_point_mean',
                  'precip_sum', 'pressure_mean', 'cloud_cover_mean', 'wind_speed_mean', 'wind_speed_max',
                  'wind_u_mean', 'wind_v_mean', 'month'] + FLAGS + ['aqi_imputed', 'city'])

pd.DataFrame({
    'logistic regression': [len(LR_NUMERIC), len(LR_BINARY), 'city (one-hot, 14)', 'yes', 'pressure_rel, precip_log, month_sin/cos'],
    'tree models': [len(TREE_FEATURES) - len(LR_BINARY) - 1, len(LR_BINARY), 'city (one-hot, 14)', 'no', 'none (raw columns)'],
}, index=['numeric features', '0/1 features', 'categorical', 'standardise', 'transforms'])
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic regression</th>
      <th>tree models</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>numeric features</th>
      <td>15</td>
      <td>19</td>
    </tr>
    <tr>
      <th>0/1 features</th>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>categorical</th>
      <td>city (one-hot, 14)</td>
      <td>city (one-hot, 14)</td>
    </tr>
    <tr>
      <th>standardise</th>
      <td>yes</td>
      <td>no</td>
    </tr>
    <tr>
      <th>transforms</th>
      <td>pressure_rel, precip_log, month_sin/cos</td>
      <td>none (raw columns)</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Final sanity checks before modelling: every column any model will use, in every split
model_cols = sorted(set(LR_NUMERIC + LR_BINARY + LR_CATEGORICAL) | set(TREE_FEATURES))

for name, part in [('train', df_train), ('val', df_val), ('test', df_test)]:
    X = part[model_cols]
    assert X.notna().all().all(), f'{name}: missing values'
    assert np.isfinite(X.select_dtypes('number')).all().all(), f'{name}: infinite values'   # city is text
    assert part[TARGET].notna().all(), f'{name}: missing target'
print(f'No missing or infinite values in the {len(model_cols)} model columns of any split.')

# Every split must contain all six classes, with enough days to score them
counts = pd.DataFrame({n: p[TARGET].value_counts() for n, p in
                       [('train', df_train), ('val', df_val), ('test', df_test)]}).reindex(AQI_CATEGORIES)
assert (counts > 0).all().all(), 'a split is missing a class'
counts
```

**Output:**

```
No missing or infinite values in the 31 model columns of any split.
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train</th>
      <th>val</th>
      <th>test</th>
    </tr>
    <tr>
      <th>tomorrow_status</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Good</th>
      <td>2990</td>
      <td>293</td>
      <td>737</td>
    </tr>
    <tr>
      <th>Satisfactory</th>
      <td>10368</td>
      <td>1785</td>
      <td>3783</td>
    </tr>
    <tr>
      <th>Moderate</th>
      <td>9826</td>
      <td>1829</td>
      <td>3428</td>
    </tr>
    <tr>
      <th>Poor</th>
      <td>5615</td>
      <td>778</td>
      <td>1446</td>
    </tr>
    <tr>
      <th>Very Poor</th>
      <td>3696</td>
      <td>368</td>
      <td>521</td>
    </tr>
    <tr>
      <th>Severe</th>
      <td>1048</td>
      <td>51</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>

**Findings:**
- Every column used by any model is complete and finite in all three splits, and every row has a target.
- All six classes appear in every split, but **the worst classes become much rarer in later years:**

| Class | Train (2015–2022) | Validation (2023) | Test (2024–2025) |
|---|---|---|---|
| Very Poor | 11.0% | 7.2% | 5.2% |
| Severe | 3.1% (1,048 days) | **1.0% (51 days)** | **0.7% (70 days)** |

**Implications for modelling:**
- **Severe recall on validation is noisy.** With 51 Severe days, each missed day moves recall by about 2 percentage points. Compare models on Severe recall only when the gap is clearly larger than that, and read it together with macro F1.
- **Training contains more bad days than later years.** Trained on a 3.1% Severe rate, with `class_weight='balanced'` pushing further, the model may over-predict Very Poor and Severe in 2023–2025. Check **precision** for those classes, not only recall.
- This is drift over time and needs a dedicated check before the final model: whether air quality is improving, or some cities' worst periods dropped out of the data.

## Summary of part 1

| Step | Decision |
|---|---|
| Dataset | `prepare_dataset`: EDA pipeline in leakage-safe order, 48,827 rows |
| Empty trend features | Drop 195 rows (0.4%) from every split, so all models are scored on the same days |
| Split | Train 2015–2022, validation 2023, test 2024–2025 |
| Transforms (logistic regression) | `pressure_rel` (rho 0.18 → 0.52), `precip_log` (skew 7.8 → 1.6), `month_sin` / `month_cos` (raw month rho 0.00 → 0.48) |
| Trend features | `aqi_today` + all four trend features is collinear (VIF ≈ 341): drop `aqi_roll7_mean` for logistic regression |
| Final LR set | 15 numeric + 7 binary + city; every VIF < 10; test dropping `humidity_mean` |
| Tree models | All raw features, all trend features, no transforms |

**Next (part 2):**
1. Persistence baseline: tomorrow's category = today's. This is the bar every model must beat.
2. Logistic regression baseline on `LR_NUMERIC` + `LR_BINARY` + one-hot `city`, standardised, `class_weight='balanced'`; scored with macro F1 and Very Poor / Severe recall on validation.
3. Variants: without `humidity_mean`, without `city`.
