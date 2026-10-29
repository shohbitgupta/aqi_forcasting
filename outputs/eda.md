<!-- Exported from notebooks/eda.ipynb: every cell run top to bottom; generated, do not edit by hand -->

# Next-Day AQI Forecasting: Exploratory Data Analysis

**Goal:** predict tomorrow's AQI category (`tomorrow_status`, six classes from *Good* to *Severe*) for 14 Indian cities, using what is known today: AQI, prominent pollutants, weather and calendar.

**Data:** `data/processed/aqi_weather_2015_2025.csv`, built by the scripts in `code/`. It joins CPCB daily AQI bulletins (Urban Emissions compilation) with Open-Meteo daily weather, May 2015 to December 2025, one row per city per day.

**Contents**
1. Setup and loading
2. Data overview and quality
3. Categorical features
4. Target: `tomorrow_status`
5. Missing values
6. Target distribution
7. Numeric feature distributions
8. Features vs target
9. Collinearity
10. Summary of decisions

Each analysis ends with a **Finding → Decision** note. Reusable helpers (target, imputation, split, box-plot statistics) live in `utility.py`.

## 1. Setup and loading

```python
%load_ext autoreload
%autoreload 2

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../scripts')                 # utility.py lives in scripts/
from utility import (AQI_CATEGORIES, aqi_to_category, ffill_aqi,
                     split_train_val_test, class_box_stats, class_signal_summary,
                     add_trend_features)

pd.set_option('display.max_columns', 40)
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True})
COLOR = '#2a6fb0'
```

```python
DATA_PATH = '../data/processed/aqi_weather_2015_2025.csv'

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.rename(columns={'aqi_value': 'aqi_today', 'air_quality_status': 'status_today'})
df = df.drop(columns=['split'])          # the split is rebuilt from `date` (split_train_val_test)
df = df.sort_values(['city', 'date']).reset_index(drop=True)
df.shape
```

**Output:**

```
(51072, 20)
```

`aqi_value` and `air_quality_status` are renamed to `aqi_today` and `status_today` to make the time direction explicit: these are **today's** values, used as inputs for **tomorrow's** prediction. The precomputed `split` column is dropped; the same time-based split is recreated from `date` before modelling.

## 2. Data overview and quality

```python
df.head()
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>date</th>
      <th>number_of_monitoring_stations</th>
      <th>prominent_pollutants</th>
      <th>aqi_today</th>
      <th>status_today</th>
      <th>temp_mean</th>
      <th>temp_max</th>
      <th>temp_min</th>
      <th>humidity_mean</th>
      <th>dew_point_mean</th>
      <th>precip_sum</th>
      <th>pressure_mean</th>
      <th>cloud_cover_mean</th>
      <th>wind_speed_mean</th>
      <th>wind_speed_max</th>
      <th>wind_u_mean</th>
      <th>wind_v_mean</th>
      <th>wind_dir_mean</th>
      <th>temp_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bengaluru</td>
      <td>2015-05-02</td>
      <td>0.0</td>
      <td>CO, PM2.5</td>
      <td>132.0</td>
      <td>Moderate</td>
      <td>25.65</td>
      <td>32.3</td>
      <td>21.5</td>
      <td>70.00</td>
      <td>19.13</td>
      <td>2.8</td>
      <td>912.20</td>
      <td>54.46</td>
      <td>2.11</td>
      <td>3.57</td>
      <td>1.79</td>
      <td>0.32</td>
      <td>259.84</td>
      <td>10.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bengaluru</td>
      <td>2015-05-03</td>
      <td>0.0</td>
      <td>CO, PM2.5</td>
      <td>49.0</td>
      <td>Good</td>
      <td>24.92</td>
      <td>31.1</td>
      <td>21.3</td>
      <td>75.17</td>
      <td>19.67</td>
      <td>6.7</td>
      <td>911.58</td>
      <td>76.17</td>
      <td>1.20</td>
      <td>2.45</td>
      <td>0.84</td>
      <td>0.10</td>
      <td>263.27</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bengaluru</td>
      <td>2015-05-04</td>
      <td>0.0</td>
      <td>CO</td>
      <td>46.0</td>
      <td>Good</td>
      <td>25.06</td>
      <td>30.8</td>
      <td>20.4</td>
      <td>72.46</td>
      <td>19.23</td>
      <td>1.8</td>
      <td>911.61</td>
      <td>60.04</td>
      <td>1.29</td>
      <td>1.99</td>
      <td>0.21</td>
      <td>0.15</td>
      <td>234.16</td>
      <td>10.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bengaluru</td>
      <td>2015-05-05</td>
      <td>0.0</td>
      <td>CO</td>
      <td>229.0</td>
      <td>Poor</td>
      <td>25.57</td>
      <td>30.8</td>
      <td>20.0</td>
      <td>64.04</td>
      <td>17.40</td>
      <td>0.0</td>
      <td>911.34</td>
      <td>77.79</td>
      <td>1.68</td>
      <td>2.52</td>
      <td>-1.29</td>
      <td>0.77</td>
      <td>120.68</td>
      <td>10.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengaluru</td>
      <td>2015-05-06</td>
      <td>0.0</td>
      <td>O3</td>
      <td>90.0</td>
      <td>Satisfactory</td>
      <td>25.22</td>
      <td>30.1</td>
      <td>21.2</td>
      <td>67.29</td>
      <td>18.34</td>
      <td>3.8</td>
      <td>910.76</td>
      <td>85.75</td>
      <td>1.17</td>
      <td>2.01</td>
      <td>-0.21</td>
      <td>0.24</td>
      <td>139.27</td>
      <td>8.9</td>
    </tr>
  </tbody>
</table>
</div>

```python
summary = pd.DataFrame({
    'dtype': df.dtypes,
    'missing': df.isnull().sum(),
    'missing_%': (df.isnull().mean() * 100).round(2),
    'unique': df.nunique(),
}).sort_values('missing', ascending=False)
summary
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dtype</th>
      <th>missing</th>
      <th>missing_%</th>
      <th>unique</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>prominent_pollutants</th>
      <td>str</td>
      <td>2139</td>
      <td>4.19</td>
      <td>52</td>
    </tr>
    <tr>
      <th>number_of_monitoring_stations</th>
      <td>float64</td>
      <td>2138</td>
      <td>4.19</td>
      <td>41</td>
    </tr>
    <tr>
      <th>aqi_today</th>
      <td>float64</td>
      <td>2138</td>
      <td>4.19</td>
      <td>487</td>
    </tr>
    <tr>
      <th>status_today</th>
      <td>str</td>
      <td>2138</td>
      <td>4.19</td>
      <td>6</td>
    </tr>
    <tr>
      <th>city</th>
      <td>str</td>
      <td>0</td>
      <td>0.00</td>
      <td>14</td>
    </tr>
    <tr>
      <th>pressure_mean</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>6893</td>
    </tr>
    <tr>
      <th>wind_dir_mean</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>24805</td>
    </tr>
    <tr>
      <th>wind_v_mean</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>1089</td>
    </tr>
    <tr>
      <th>wind_u_mean</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>1382</td>
    </tr>
    <tr>
      <th>wind_speed_max</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>929</td>
    </tr>
    <tr>
      <th>wind_speed_mean</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>751</td>
    </tr>
    <tr>
      <th>cloud_cover_mean</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>2401</td>
    </tr>
    <tr>
      <th>dew_point_mean</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>2669</td>
    </tr>
    <tr>
      <th>precip_sum</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>698</td>
    </tr>
    <tr>
      <th>date</th>
      <td>datetime64[us]</td>
      <td>0</td>
      <td>0.00</td>
      <td>3898</td>
    </tr>
    <tr>
      <th>humidity_mean</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>1972</td>
    </tr>
    <tr>
      <th>temp_min</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>333</td>
    </tr>
    <tr>
      <th>temp_max</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>341</td>
    </tr>
    <tr>
      <th>temp_mean</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>2974</td>
    </tr>
    <tr>
      <th>temp_range</th>
      <td>float64</td>
      <td>0</td>
      <td>0.00</td>
      <td>199</td>
    </tr>
  </tbody>
</table>
</div>

**Finding:** all weather columns are complete. The four bulletin columns (`aqi_today`, `status_today`, `number_of_monitoring_stations`, `prominent_pollutants`) are missing together on about 4.2% of days, the days with no CPCB bulletin.
**Decision:** handle these rows in section 5, after the target is built.

```python
print('Duplicate (city, date) rows:', df.duplicated(['city', 'date']).sum())

coverage = (df.groupby('city')
              .agg(first_day=('date', 'min'), last_day=('date', 'max'), days=('date', 'size'),
                   missing_aqi_pct=('aqi_today', lambda s: s.isna().mean() * 100),
                   mean_aqi=('aqi_today', 'mean'))
              .round({'missing_aqi_pct': 1, 'mean_aqi': 1})
              .sort_values('mean_aqi', ascending=False))
coverage
```

**Output:**

```
Duplicate (city, date) rows: 0
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_day</th>
      <th>last_day</th>
      <th>days</th>
      <th>missing_aqi_pct</th>
      <th>mean_aqi</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Delhi</th>
      <td>2015-05-01</td>
      <td>2025-12-31</td>
      <td>3898</td>
      <td>0.6</td>
      <td>214.8</td>
    </tr>
    <tr>
      <th>Ghaziabad</th>
      <td>2017-05-03</td>
      <td>2025-12-31</td>
      <td>3165</td>
      <td>0.5</td>
      <td>212.3</td>
    </tr>
    <tr>
      <th>Greater Noida</th>
      <td>2018-05-26</td>
      <td>2025-12-31</td>
      <td>2777</td>
      <td>0.5</td>
      <td>206.5</td>
    </tr>
    <tr>
      <th>Faridabad</th>
      <td>2015-05-01</td>
      <td>2025-12-31</td>
      <td>3898</td>
      <td>10.9</td>
      <td>200.3</td>
    </tr>
    <tr>
      <th>Noida</th>
      <td>2017-05-04</td>
      <td>2025-12-31</td>
      <td>3164</td>
      <td>0.9</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>Gurugram</th>
      <td>2015-12-15</td>
      <td>2025-12-31</td>
      <td>3670</td>
      <td>8.6</td>
      <td>196.1</td>
    </tr>
    <tr>
      <th>Patna</th>
      <td>2015-09-18</td>
      <td>2025-12-31</td>
      <td>3758</td>
      <td>11.7</td>
      <td>181.8</td>
    </tr>
    <tr>
      <th>Lucknow</th>
      <td>2015-05-02</td>
      <td>2025-12-31</td>
      <td>3897</td>
      <td>2.8</td>
      <td>170.4</td>
    </tr>
    <tr>
      <th>Kanpur</th>
      <td>2015-05-01</td>
      <td>2025-12-31</td>
      <td>3898</td>
      <td>4.9</td>
      <td>154.1</td>
    </tr>
    <tr>
      <th>Kolkata</th>
      <td>2016-07-05</td>
      <td>2025-12-31</td>
      <td>3467</td>
      <td>6.5</td>
      <td>112.7</td>
    </tr>
    <tr>
      <th>Mumbai</th>
      <td>2015-08-20</td>
      <td>2025-12-31</td>
      <td>3787</td>
      <td>4.6</td>
      <td>104.5</td>
    </tr>
    <tr>
      <th>Hyderabad</th>
      <td>2015-05-01</td>
      <td>2025-12-31</td>
      <td>3898</td>
      <td>0.9</td>
      <td>89.4</td>
    </tr>
    <tr>
      <th>Chennai</th>
      <td>2015-05-01</td>
      <td>2025-12-31</td>
      <td>3898</td>
      <td>2.4</td>
      <td>84.5</td>
    </tr>
    <tr>
      <th>Bengaluru</th>
      <td>2015-05-02</td>
      <td>2025-12-31</td>
      <td>3897</td>
      <td>1.2</td>
      <td>75.2</td>
    </tr>
  </tbody>
</table>
</div>

**Finding:** there are no duplicate city-days, so `shift()` and rolling windows are safe. Coverage is uneven: Ghaziabad and Noida start in 2017 and Greater Noida in 2018, so the early years are dominated by the other cities. Average AQI ranges from about 75 (Bengaluru) to about 215 (Delhi). Missing days are concentrated in Patna (11.7%), Faridabad (10.9%) and Gurugram (8.6%); most other cities miss under 5%.
**Decision:** keep all cities; the city-level differences are examined in section 3.1, and the gaps in section 5.

### 2.1 Label check

The AQI category should follow from the AQI value through the CPCB ranges (0–50 Good, 51–100 Satisfactory, 101–200 Moderate, 201–300 Poor, 301–400 Very Poor, 401–500 Severe).

```python
known = df.dropna(subset=['aqi_today'])
from_value = aqi_to_category(known['aqi_today']).astype(str).rename('category from aqi_today')
mismatch = known[from_value != known['status_today']]

print(f'Rows whose label disagrees with their AQI value: {len(mismatch)}')
print('By year:', mismatch['date'].dt.year.value_counts().sort_index().to_dict())
pd.crosstab(mismatch['status_today'], from_value[mismatch.index])
```

**Output:**

```
Rows whose label disagrees with their AQI value: 56
By year: {2015: 4, 2018: 51, 2023: 1}
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>category from aqi_today</th>
      <th>Good</th>
      <th>Moderate</th>
      <th>Poor</th>
      <th>Satisfactory</th>
      <th>Very Poor</th>
    </tr>
    <tr>
      <th>status_today</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Moderate</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Poor</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Satisfactory</th>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Severe</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Very Poor</th>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

**Finding:** 56 rows carry a label that does not match their AQI value, almost all in 2018 (for example *Moderate* with an AQI in the *Satisfactory* range).
**Decision:** never use the text label as the target. Build `tomorrow_status` from the AQI **value** with `aqi_to_category`, and recompute `status_today` the same way (`ffill_aqi` does this).

## 3. Categorical features

```python
categorical = df.select_dtypes(include=['object', 'string']).columns.tolist()
df[categorical].nunique()
```

**Output:**

```
city                    14
prominent_pollutants    52
status_today             6
dtype: int64
```

- `status_today` becomes the basis of the target (section 4).
- `city` is nominal (no order).
- `prominent_pollutants` is a comma-separated, multi-label field.

### 3.1 City

```python
city_mix = (pd.crosstab(df['city'], df['status_today'], normalize='index') * 100).reindex(columns=AQI_CATEGORIES)
city_mix.round(0).sort_values('Good')
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>status_today</th>
      <th>Good</th>
      <th>Satisfactory</th>
      <th>Moderate</th>
      <th>Poor</th>
      <th>Very Poor</th>
      <th>Severe</th>
    </tr>
    <tr>
      <th>city</th>
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
      <th>Delhi</th>
      <td>0.0</td>
      <td>17.0</td>
      <td>32.0</td>
      <td>27.0</td>
      <td>19.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Ghaziabad</th>
      <td>2.0</td>
      <td>17.0</td>
      <td>32.0</td>
      <td>25.0</td>
      <td>18.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Greater Noida</th>
      <td>2.0</td>
      <td>16.0</td>
      <td>35.0</td>
      <td>27.0</td>
      <td>16.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Noida</th>
      <td>2.0</td>
      <td>19.0</td>
      <td>35.0</td>
      <td>23.0</td>
      <td>16.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Faridabad</th>
      <td>3.0</td>
      <td>18.0</td>
      <td>34.0</td>
      <td>26.0</td>
      <td>15.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Gurugram</th>
      <td>3.0</td>
      <td>17.0</td>
      <td>34.0</td>
      <td>29.0</td>
      <td>15.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Patna</th>
      <td>3.0</td>
      <td>26.0</td>
      <td>31.0</td>
      <td>24.0</td>
      <td>13.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Lucknow</th>
      <td>4.0</td>
      <td>28.0</td>
      <td>35.0</td>
      <td>20.0</td>
      <td>12.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Kanpur</th>
      <td>8.0</td>
      <td>33.0</td>
      <td>33.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Chennai</th>
      <td>11.0</td>
      <td>64.0</td>
      <td>23.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Mumbai</th>
      <td>13.0</td>
      <td>43.0</td>
      <td>37.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Hyderabad</th>
      <td>15.0</td>
      <td>50.0</td>
      <td>34.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Bengaluru</th>
      <td>19.0</td>
      <td>63.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Kolkata</th>
      <td>26.0</td>
      <td>34.0</td>
      <td>24.0</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

**Finding:** cities behave very differently. Delhi NCR cities spend about 45–50% of days at *Poor* or worse, while Bengaluru, Chennai and Hyderabad almost never do. Cities with similar climates also differ: Patna (about 40% *Poor* or worse) is much worse than Kolkata (about 17%). Weather alone does not explain this; `city` stands in for emissions and geography, which the dataset does not measure.
**Decision:** keep `city` and one-hot encode it (14 columns). Once lag features exist, compare validation scores with and without `city`, since recent AQI history may already capture the city baseline.

### 3.2 Prominent pollutants

```python
pollutants = df['prominent_pollutants'].dropna().str.split(',').explode().str.strip()
print('Distinct strings:', df['prominent_pollutants'].nunique())
print('Pollutants listed per day:', df['prominent_pollutants'].dropna().str.count(',').add(1).value_counts().sort_index().to_dict())
pollutants.value_counts()
```

**Output:**

```
Distinct strings: 52
Pollutants listed per day: {1: 24874, 2: 18565, 3: 5493, 4: 1}
```

**Output:**

```
prominent_pollutants
PM2.5    32009
PM10     26971
O3        8366
CO        7684
NO2       2985
SO2        458
NH3         14
Name: count, dtype: int64
```

```python
flags = df['prominent_pollutants'].str.get_dummies(sep=', ')
flags.columns = 'has_' + flags.columns.str.lower().str.replace('.', '', regex=False)

df = pd.concat([df, flags], axis=1)
df = df.drop(columns=['prominent_pollutants', 'has_nh3'])   # NH3: only 14 days, too rare to learn from

flag_cols = [c for c in df.columns if c.startswith('has_')]
df[flag_cols].sum()
```

**Output:**

```
has_co       7684
has_no2      2985
has_o3       8366
has_pm10    26971
has_pm25    32009
has_so2       458
dtype: int64
```

**Finding:** the 52 distinct strings are combinations of 7 pollutants, and about half of all days list two or three. One-hot encoding the raw strings would create many rare columns.
**Decision:** turn the field into one 0/1 flag per pollutant and drop `has_nh3` (14 days). On days with no bulletin, every flag is 0; `aqi_imputed` (section 5) tells the model those flags are unknown.

## 4. Target: `tomorrow_status`

The model predicts **tomorrow's** category from **today's** data, so each row needs tomorrow's category as its label. `groupby('city')['aqi_today'].shift(-1)` moves each city's AQI up by one row, so every day receives the next day's AQI. `aqi_to_category` then converts it into the CPCB category.

| Rule | Why |
|---|---|
| Shift **within each city** | Bengaluru's last day must not receive Chennai's first day as "tomorrow" |
| Build from the AQI **value** | Avoids the 56 wrong labels found in section 2.1 |
| Build it **before** filling gaps | A filled (copied) AQI must never become an answer the model learns from |
| Features look back (`shift(+n)`), the target looks forward (`shift(-1)`) | Any feature that looks forward leaks the answer |

The last day of each city has no tomorrow, so its target is empty and the row is dropped in section 5.

```python
df['tomorrow_status'] = aqi_to_category(df.groupby('city')['aqi_today'].shift(-1))
df['tomorrow_status'].value_counts(sort=False)
```

**Output:**

```
tomorrow_status
Good             4043
Satisfactory    16027
Moderate        15161
Poor             7890
Very Poor        4618
Severe           1181
Name: count, dtype: int64
```

```python
# Sanity check: each row's target is the next day's category (Delhi, autumn 2023)
df.loc[(df['city'] == 'Delhi') & df['date'].between('2023-10-31', '2023-11-04'),
       ['city', 'date', 'aqi_today', 'status_today', 'tomorrow_status']]
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
      <th>status_today</th>
      <th>tomorrow_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10900</th>
      <td>Delhi</td>
      <td>2023-10-31</td>
      <td>359.0</td>
      <td>Very Poor</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>10901</th>
      <td>Delhi</td>
      <td>2023-11-01</td>
      <td>364.0</td>
      <td>Very Poor</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>10902</th>
      <td>Delhi</td>
      <td>2023-11-02</td>
      <td>392.0</td>
      <td>Very Poor</td>
      <td>Severe</td>
    </tr>
    <tr>
      <th>10903</th>
      <td>Delhi</td>
      <td>2023-11-03</td>
      <td>468.0</td>
      <td>Severe</td>
      <td>Severe</td>
    </tr>
    <tr>
      <th>10904</th>
      <td>Delhi</td>
      <td>2023-11-04</td>
      <td>415.0</td>
      <td>Severe</td>
      <td>Severe</td>
    </tr>
  </tbody>
</table>
</div>

On 2 November, today is *Very Poor* (392) but tomorrow is *Severe* (468). A "tomorrow equals today" guess misses exactly this kind of jump, which is where the model can add value.

## 5. Missing values

About 4.2% of days have no bulletin. The options were compared by hiding 3,000 known AQI values and filling them back:

| Method | Mean error (AQI points) | Category correct |
|---|---|---|
| Fill with 0 | 154 | 8% |
| Global mean | 83 | 32% |
| City mean | 67 | 37% |
| City and month mean | 43 | 52% |
| **Forward fill (last known value, same city)** | **29** | **68%** |
| Linear interpolation | 22 | 74%, but it uses tomorrow's value: **leakage** |

Forward fill is the most accurate method that uses only the past. First, check how long the gaps are:

```python
missing = df['aqi_today'].isna()
gap_id = (missing != missing.groupby(df['city']).shift()).cumsum()
gaps = missing.groupby([df['city'], gap_id]).agg(['all', 'size'])
gap_len = gaps.loc[gaps['all'], 'size']

print(f'Gaps: {len(gap_len)} | 1 day: {(gap_len == 1).sum()} | '
      f'3 days or shorter: {(gap_len <= 3).mean():.0%} | longest: {gap_len.max()} days')
gap_len.value_counts().sort_index().head(10)
```

**Output:**

```
Gaps: 903 | 1 day: 577 | 3 days or shorter: 90% | longest: 177 days
```

**Output:**

```
size
1     577
2     158
3      75
4      29
5      17
6      12
7       4
8       5
9       5
10      5
Name: count, dtype: int64
```

```python
print('Missing aqi_today before:', df['aqi_today'].isna().sum())

df = ffill_aqi(df, limit=3)       # fill gaps of up to 3 days; adds aqi_imputed, recomputes status_today
print('Filled (gaps of 3 days or shorter):', df['aqi_imputed'].sum())
print('Still missing (longer gaps):', df['aqi_today'].isna().sum())

rows_before = len(df)
df = df.dropna(subset=['aqi_today', 'tomorrow_status'])               # rows need a real input AND a real answer
df = df.drop(columns=['number_of_monitoring_stations'])                # describes the network, not the air
df = df.reset_index(drop=True)
print(f'Rows: {rows_before:,} -> {len(df):,}')
print('Missing values left:', df.isnull().sum().sum())
df.shape
```

**Output:**

```
Missing aqi_today before: 2138
Filled (gaps of 3 days or shorter): 1397
Still missing (longer gaps): 741
Rows: 51,072 -> 48,827
Missing values left: 0
```

**Output:**

```
(48827, 26)
```

**Finding:** 90% of gaps are 3 days or shorter (577 of the 903 gaps last a single day); 1,397 values are filled, and the 741 days in longer gaps are dropped, together with rows that have no real next-day AQI. In total 2,245 rows (4.4%) are removed, leaving 48,827 rows with no missing values. The class balance hardly changes.
**Decision:**
- Inputs may be filled; the target never is.
- `aqi_imputed = 1` marks filled days so the model can trust those values less.
- `number_of_monitoring_stations` is dropped: it has no relationship with AQI (correlation ≈ 0) and is missing whenever a day is filled.

## 6. Target distribution

```python
target = 'tomorrow_status'
counts = df[target].value_counts().reindex(AQI_CATEGORIES)
share = counts / counts.sum()

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(counts.index, counts.values, color=COLOR, linewidth=2, marker='o', markersize=8)
for x, v in zip(counts.index, counts.values):
    ax.annotate(f'{v:,}\n({v / counts.sum():.1%})', (x, v), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9)
ax.annotate(f"Severe: {counts['Satisfactory'] / counts['Severe']:.1f}x fewer days than Satisfactory.\n"
            "A model could ignore it and still look accurate\n"
            "→ score with macro F1 and Severe recall",
            ('Severe', counts['Severe']), xytext=(3.35, counts.max() * 0.7), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='grey'))
ax.annotate(f"Satisfactory + Moderate:\n{share['Satisfactory'] + share['Moderate']:.0%} of all days",
            (1.5, counts.max() * 0.72), ha='center', fontsize=9, color='#555555')
fig.suptitle(f"The target is imbalanced: Severe is only {share['Severe']:.1%} of days",
             x=0.01, ha='left', fontsize=12, fontweight='bold')
ax.set_title('Days per category of tomorrow_status (all cities, 2015–2025)', loc='left', fontsize=10)
ax.set_ylabel('Days')
ax.set_ylim(0, counts.max() * 1.3)
plt.tight_layout()
plt.show()
```

![chart](eda_files/eda_31_1.png)

**Finding:** the classes are imbalanced. Satisfactory (32.8%) has about 13.6 times as many days as Severe (2.4%), and Satisfactory plus Moderate make up about two thirds of the data. A model could ignore Severe entirely and still reach a high accuracy.
**Decision:**
- Evaluate with **macro F1**, which gives every class equal weight, plus **recall on Very Poor and Severe**, the days that matter most.
- Train with `class_weight='balanced'` (weight = rows / (6 × rows in class)), which makes each Severe day count about 7 times more and each Satisfactory day about half.

```python
share = (pd.crosstab(df['date'].dt.month, df[target], normalize='index')
           .reindex(columns=AQI_CATEGORIES) * 100)
colors = plt.cm.YlOrRd(np.linspace(0.25, 1.0, len(AQI_CATEGORIES)))   # light = Good, dark = Severe
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(figsize=(11, 5.5))
for start, end in [(0.5, 1.5), (10.5, 12.5)]:                              # winter: Nov–Jan
    ax.axvspan(start, end, color='#f4dcd6', zorder=0)
ax.axvspan(6.5, 9.5, color='#dbe9f6', zorder=0)                            # monsoon: Jul–Sep
ax.text(8, 66, 'Monsoon: rain cleans the air,\nalmost no Poor or worse days', ha='center', fontsize=9)
ax.text(11.5, 66, 'Winter: the worst air', ha='center', fontsize=9)
for cat, c in zip(AQI_CATEGORIES, colors):
    ax.plot(share.index, share[cat], color=c, linewidth=2, marker='o', markersize=5, label=cat)
peak = share['Severe'].idxmax()
ax.annotate(f"Severe peaks in {MONTHS[peak - 1]}:\n{share.loc[peak, 'Severe']:.1f}% of days",
            (peak, share.loc[peak, 'Severe']), xytext=(9.6, 42), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='grey'))
ax.set_xticks(range(1, 13), MONTHS)
ax.set_xlim(0.5, 12.5)
ax.set_ylim(-2, 74)
ax.set_ylabel('% of days')
fig.suptitle('Strong seasonality: winter is the worst, the monsoon the cleanest  → add month as a feature',
             x=0.01, ha='left', fontsize=12, fontweight='bold')
ax.set_title('Share of days in each category of tomorrow_status, by month (all cities, 2015–2025)',
             loc='left', fontsize=10)
ax.legend(title='Status', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.show()
```

![chart](eda_files/eda_33_2.png)

**Finding:** strong seasonality. The monsoon (July–September) is mostly Satisfactory or Good, with almost no Poor days. Air quality worsens from October (stubble burning, cooler nights), and November to January is the worst period, with Severe peaking at about 10% of days in November. Severe days occur almost only in winter.
**Decision:** add `month` (and possibly a winter or stubble-season flag) as a feature. Expect Severe recall to be the hardest metric, because the model learns it from a few winter months.

## 7. Numeric feature distributions

A histogram shows one feature on its own: its shape, skew, unusual peaks and impossible values. Skew above about ±1 means a long tail, which can distort logistic regression.

```python
numerical = [c for c in df.select_dtypes(include='number').columns
             if not c.startswith('has_') and c != 'aqi_imputed']
LONG_TAIL = '#d9822b'
notes = {                                                   # (text, x, horizontal alignment)
    'precip_sum': ('Most days are dry; a few have\nheavy rain → log1p for\nlogistic regression', 0.97, 'right'),
    'pressure_mean': ('3 peaks = city altitude\n(Bengaluru, Hyderabad)\n→ pressure vs city normal', 0.03, 'left'),
    'cloud_cover_mean': ('Peaks at 0 and 100%:\nclear or overcast days', 0.5, 'center'),
}

ncols = 4
nrows = int(np.ceil(len(numerical) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
for ax, col in zip(axes.flat, numerical):
    skew = df[col].skew()
    long_tail = round(abs(skew), 1) > 1
    ax.hist(df[col], bins=50, color=LONG_TAIL if long_tail else COLOR)
    ax.set_title(f'{col}  (skew={skew:.1f})', fontsize=10, fontweight='bold' if long_tail else 'normal')
    if col in notes:
        text, x, ha = notes[col]
        ax.text(x, 0.95, text, transform=ax.transAxes, ha=ha, va='top', fontsize=8.5,
                bbox=dict(facecolor='white', edgecolor='#999999', boxstyle='round,pad=0.3'))
for ax in axes.flat[len(numerical):]:
    ax.set_visible(False)
fig.suptitle('Values are plausible; orange = long tail (|skew| > 1). The notes mark what needs a fix',
             x=0.01, ha='left', fontsize=13, fontweight='bold')
plt.tight_layout(rect=(0, 0, 1, 0.98))
plt.show()
```

![chart](eda_files/eda_36_3.png)

```python
# Outliers by the IQR rule (outside Q1 - 1.5*IQR .. Q3 + 1.5*IQR), with the observed range
q1, q3 = df[numerical].quantile(0.25), df[numerical].quantile(0.75)
iqr = q3 - q1
outliers = ((df[numerical] < q1 - 1.5 * iqr) | (df[numerical] > q3 + 1.5 * iqr)).sum()

pd.DataFrame({'outliers': outliers,
              'pct': (outliers / len(df) * 100).round(1),
              'min': df[numerical].min(),
              'max': df[numerical].max()}).sort_values('outliers', ascending=False)
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>outliers</th>
      <th>pct</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precip_sum</th>
      <td>8427</td>
      <td>17.3</td>
      <td>0.00</td>
      <td>251.10</td>
    </tr>
    <tr>
      <th>pressure_mean</th>
      <td>3849</td>
      <td>7.9</td>
      <td>904.52</td>
      <td>1018.52</td>
    </tr>
    <tr>
      <th>wind_speed_mean</th>
      <td>1490</td>
      <td>3.1</td>
      <td>0.36</td>
      <td>12.22</td>
    </tr>
    <tr>
      <th>wind_v_mean</th>
      <td>1444</td>
      <td>3.0</td>
      <td>-8.46</td>
      <td>9.27</td>
    </tr>
    <tr>
      <th>humidity_mean</th>
      <td>986</td>
      <td>2.0</td>
      <td>10.08</td>
      <td>97.92</td>
    </tr>
    <tr>
      <th>temp_max</th>
      <td>844</td>
      <td>1.7</td>
      <td>12.20</td>
      <td>47.40</td>
    </tr>
    <tr>
      <th>wind_speed_max</th>
      <td>605</td>
      <td>1.2</td>
      <td>0.68</td>
      <td>19.14</td>
    </tr>
    <tr>
      <th>temp_mean</th>
      <td>578</td>
      <td>1.2</td>
      <td>6.19</td>
      <td>40.28</td>
    </tr>
    <tr>
      <th>aqi_today</th>
      <td>548</td>
      <td>1.1</td>
      <td>14.00</td>
      <td>500.00</td>
    </tr>
    <tr>
      <th>wind_u_mean</th>
      <td>262</td>
      <td>0.5</td>
      <td>-8.57</td>
      <td>8.61</td>
    </tr>
    <tr>
      <th>temp_min</th>
      <td>38</td>
      <td>0.1</td>
      <td>1.00</td>
      <td>35.60</td>
    </tr>
    <tr>
      <th>dew_point_mean</th>
      <td>0</td>
      <td>0.0</td>
      <td>-4.30</td>
      <td>28.55</td>
    </tr>
    <tr>
      <th>cloud_cover_mean</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>wind_dir_mean</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.05</td>
      <td>359.96</td>
    </tr>
    <tr>
      <th>temp_range</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>20.70</td>
    </tr>
  </tbody>
</table>
</div>

```python
# pressure_mean has three separate peaks: check whether they come from the cities
df.groupby('city')['pressure_mean'].median().sort_values().round(0)
```

**Output:**

```
city
Bengaluru         912.0
Hyderabad         953.0
Gurugram          982.0
Ghaziabad         983.0
Delhi             984.0
Faridabad         985.0
Noida             985.0
Greater Noida     986.0
Kanpur            992.0
Lucknow           994.0
Patna            1001.0
Kolkata          1006.0
Chennai          1007.0
Mumbai           1009.0
Name: pressure_mean, dtype: float64
```

**Findings:**
- `precip_sum` is very skewed (skew ≈ 7.6): about 0 on most days, up to 251 mm. Its 17% "outliers" are simply rainy days.
- `pressure_mean` has three peaks. This is **altitude**, not weather: the city medians are 912 hPa in Bengaluru and 953 hPa in Hyderabad, which sit higher, against 982–1009 hPa everywhere else. On its own, the feature mostly identifies the city.
- All values are physically plausible (humidity 10–98%, temperature 1–47 °C, AQI 14–500). The extremes are real events, including the Severe days the model most needs to learn.

**Decisions:**
- Remove no outliers.
- For logistic regression, use `np.log1p(precip_sum)` or a `rained` 0/1 flag. Tree models need no transformation.
- Replace `pressure_mean` with pressure relative to each city's normal (`pressure_mean` minus the city's **training-set** mean), or drop it. Decide on validation.

## 8. Features vs target

A box plot per class shows how today's value of a feature relates to **tomorrow's** category. The box holds the middle 50% of days and the line is the median. Four questions read the signal:

1. **Do the medians move in a clear direction?** Yes means the feature carries signal.
2. **Do the boxes overlap?** Heavy overlap means the feature alone cannot separate those classes.
3. **Is the pattern straight or does it bend?** Logistic regression can only follow straight-line patterns; trees can follow bends.
4. **Which classes look the same?** Those classes need other features to tell them apart.

```python
features = ['aqi_today', 'temp_min', 'temp_range', 'dew_point_mean', 'humidity_mean',
            'wind_speed_mean', 'cloud_cover_mean', 'precip_sum', 'pressure_mean']
signal = class_signal_summary(df, features)       # the numbers behind each panel (explained below)

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
for ax, col in zip(axes.flat, features):
    s = signal.loc[col]
    strong = abs(s['rho']) >= 0.3
    groups = [df.loc[df[target] == c, col] for c in AQI_CATEGORIES]
    ax.boxplot(groups, tick_labels=AQI_CATEGORIES, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='#cfe0f1' if strong else '#e6e6e6', color=COLOR if strong else 'grey'),
               medianprops=dict(color='#0b3d6e', linewidth=2))
    title = f"{col}\nrho {s['rho']:+.2f} ({s['direction']}), shape: {s['shape']}"
    if s['same_as'] != '-':
        title += f"\nlooks the same: {s['same_as']}"
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', rotation=30)
fig.suptitle("aqi_today separates tomorrow's classes best. Weather tells clean from polluted days, "
             "but not Very Poor from Severe\n(blue = strong signal, |rho| ≥ 0.3; grey = weak; outliers hidden)",
             x=0.01, ha='left', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

![chart](eda_files/eda_41_4.png)

`class_signal_summary` (in `utility.py`) turns the box plots into numbers, one row per feature. How to read each column:

| Column | Plain question | How to read it | Example |
|---|---|---|---|
| `rho` | Does the feature go up or down as tomorrow gets worse? | A score from −1 to +1. **Near 0** = no link. **Plus** = the value rises as air gets worse, **minus** = it falls. The further from 0, the stronger the link. | `aqi_today` 0.88 (strong); `temp_min` −0.51 (colder nights before worse air); `pressure_mean` 0.16 (weak) |
| `direction` | The same answer in words | `up` / `down` (strong, \|rho\| ≥ 0.3), `weak up` / `weak down` (0.1–0.3), `none` (below 0.1) | |
| `medians` | What is the typical value in each class? | The middle line of each box, from Good to Severe | `temp_min`: 25 → 25 → 22 → 17.5 → 12 → 12 |
| `avg_box_overlap` | Do neighbouring boxes sit on top of each other? | **0** = boxes apart, the feature tells the classes apart easily. **1** = one box fully inside the other, it cannot. Averaged over the five neighbouring pairs (Good–Satisfactory, Satisfactory–Moderate, …). | `aqi_today` 0.00; `temp_min` 0.80 |
| `good_vs_severe_overlap` | Can it at least tell the best days from the worst? | Same 0-to-1 scale, but only Good against Severe | `pressure_mean` 1.0: Good and Severe days have the same pressure |
| `linear_r2` | How close are the six medians to a straight line? | **1** = perfectly straight; lower = more curved | `aqi_today` 0.98 |
| `shape` | What does the line of medians look like? | **straight**: a steady rise or fall, any model can use it. **bends**: it moves, then flattens. **reverses**: it goes down and then up (a U-shape), which logistic regression cannot use. | `aqi_today` straight; `temp_min` bends; `humidity_mean` reverses (84 → 63 → 70) |
| `same_as` | Which neighbouring classes look the same? | Two classes are listed when their medians are almost equal **and** their boxes mostly overlap: this feature cannot tell them apart. | `temp_min`: Very Poor≈Severe (both 12 °C) |

**Reading a row as a sentence.** `temp_min`: *colder nights come before worse air (`rho` −0.51, `down`), but the boxes overlap a lot (0.80), so temperature cannot decide on its own. The drop flattens at the bad end (`bends`), and it cannot tell Very Poor from Severe (`same_as`).*

These are rules of thumb, not statistical tests. The cut-offs (for example 0.3 for a strong `rho`) can be changed in the function. `rho` is a Spearman correlation: it only uses the *order* of the values, which suits ordered classes like Good → Severe.

```python
class_signal_summary(df, features)
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
      <th>avg_box_overlap</th>
      <th>good_vs_severe_overlap</th>
      <th>linear_r2</th>
      <th>shape</th>
      <th>same_as</th>
    </tr>
    <tr>
      <th>feature</th>
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
      <th>aqi_today</th>
      <td>0.88</td>
      <td>up</td>
      <td>46 → 73 → 139 → 236 → 332 → 406</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.98</td>
      <td>straight</td>
      <td>-</td>
    </tr>
    <tr>
      <th>temp_range</th>
      <td>0.63</td>
      <td>up</td>
      <td>5.7 → 7.3 → 10.8 → 12.5 → 12.7 → 13.2</td>
      <td>0.64</td>
      <td>0.0</td>
      <td>0.88</td>
      <td>bends</td>
      <td>Poor≈Very Poor</td>
    </tr>
    <tr>
      <th>dew_point_mean</th>
      <td>-0.63</td>
      <td>down</td>
      <td>24.2 → 23.3 → 16.8 → 12.3 → 10.9 → 11.1</td>
      <td>0.77</td>
      <td>0.0</td>
      <td>0.89</td>
      <td>bends</td>
      <td>Very Poor≈Severe</td>
    </tr>
    <tr>
      <th>precip_sum</th>
      <td>-0.60</td>
      <td>down</td>
      <td>5.9 → 1.1 → 0 → 0 → 0 → 0</td>
      <td>0.95</td>
      <td>0.0</td>
      <td>0.55</td>
      <td>bends</td>
      <td>Moderate≈Poor, Poor≈Very Poor, Very Poor≈Severe</td>
    </tr>
    <tr>
      <th>cloud_cover_mean</th>
      <td>-0.59</td>
      <td>down</td>
      <td>93.5 → 69.2 → 22 → 9 → 7.4 → 5.4</td>
      <td>0.78</td>
      <td>0.0</td>
      <td>0.82</td>
      <td>bends</td>
      <td>Poor≈Very Poor, Very Poor≈Severe</td>
    </tr>
    <tr>
      <th>temp_min</th>
      <td>-0.51</td>
      <td>down</td>
      <td>25.1 → 25.1 → 22.2 → 17.5 → 12 → 12</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>0.93</td>
      <td>bends</td>
      <td>Good≈Satisfactory, Very Poor≈Severe</td>
    </tr>
    <tr>
      <th>humidity_mean</th>
      <td>-0.40</td>
      <td>down</td>
      <td>84.5 → 77.3 → 62.8 → 63 → 69.2 → 69.8</td>
      <td>0.73</td>
      <td>0.0</td>
      <td>0.38</td>
      <td>reverses</td>
      <td>Moderate≈Poor, Very Poor≈Severe</td>
    </tr>
    <tr>
      <th>wind_speed_mean</th>
      <td>-0.32</td>
      <td>down</td>
      <td>3.4 → 2.8 → 2.5 → 2.4 → 2 → 1.9</td>
      <td>0.82</td>
      <td>0.0</td>
      <td>0.94</td>
      <td>bends</td>
      <td>-</td>
    </tr>
    <tr>
      <th>pressure_mean</th>
      <td>0.16</td>
      <td>weak up</td>
      <td>987.9 → 985.3 → 987.5 → 990.4 → 992.2 → 991.9</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>0.73</td>
      <td>reverses</td>
      <td>Very Poor≈Severe</td>
    </tr>
  </tbody>
</table>
</div>

```python
# The numbers behind the box plots, for checking by hand
class_box_stats(df, ['aqi_today', 'temp_min'])
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tomorrow_status</th>
      <th>Good</th>
      <th>Satisfactory</th>
      <th>Moderate</th>
      <th>Poor</th>
      <th>Very Poor</th>
      <th>Severe</th>
    </tr>
    <tr>
      <th></th>
      <th>stat</th>
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
      <th rowspan="3" valign="top">aqi_today</th>
      <th>q1</th>
      <td>39.0</td>
      <td>60.0</td>
      <td>110.0</td>
      <td>196.25</td>
      <td>294.0</td>
      <td>372.00</td>
    </tr>
    <tr>
      <th>median</th>
      <td>46.0</td>
      <td>73.0</td>
      <td>139.0</td>
      <td>236.00</td>
      <td>332.0</td>
      <td>406.00</td>
    </tr>
    <tr>
      <th>q3</th>
      <td>55.0</td>
      <td>91.0</td>
      <td>175.0</td>
      <td>277.00</td>
      <td>368.0</td>
      <td>438.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">temp_min</th>
      <th>q1</th>
      <td>22.5</td>
      <td>21.7</td>
      <td>17.4</td>
      <td>11.90</td>
      <td>8.7</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>median</th>
      <td>25.1</td>
      <td>25.1</td>
      <td>22.2</td>
      <td>17.50</td>
      <td>12.0</td>
      <td>12.00</td>
    </tr>
    <tr>
      <th>q3</th>
      <td>26.1</td>
      <td>26.3</td>
      <td>25.7</td>
      <td>22.40</td>
      <td>16.7</td>
      <td>15.38</td>
    </tr>
  </tbody>
</table>
</div>

**Findings:**
- `aqi_today` is by far the strongest signal (rho = 0.88). It is the only feature whose boxes do not overlap at all, with a straight, steady rise.
- `temp_min`, `temp_range`, `dew_point_mean`, `cloud_cover_mean` and `precip_sum` are strong (|rho| about 0.5–0.6). Cold, dry, clear and calm days come before bad air, as the inversion physics predicts.
- The weather features **bend**: they change a lot between Good and Poor, then flatten. **Very Poor and Severe look the same in almost every weather feature.** Weather signals that a bad day is coming, but not how bad; only `aqi_today` separates them clearly.
- `humidity_mean` **reverses** (U-shape, lowest at Moderate and Poor), which logistic regression cannot use.
- `pressure_mean` shows almost no signal (rho = 0.16; the Severe box lies entirely inside the Good box), consistent with the altitude effect in section 7.

**Decisions:**
- Keep `aqi_today` and the strong weather features.
- Add lag and rolling features (section 10) so the model can see the recent trend, not only today.
- The bends and the U-shape suggest that tree models (random forest, XGBoost) will beat logistic regression; logistic regression remains the interpretable baseline.

```python
flag_share = (df.groupby(target, observed=True)[flag_cols].mean() * 100).reindex(AQI_CATEGORIES)

fig, ax = plt.subplots(figsize=(10, 4.8))
for col in flag_cols:
    particle = col in ('has_pm25', 'has_pm10')
    ax.plot(range(len(AQI_CATEGORIES)), flag_share[col], marker='o', label=col,
            linewidth=2.5 if particle else 1.5, alpha=1 if particle else 0.7)
pm25 = flag_share['has_pm25']
ax.annotate(f"PM2.5 is prominent before {pm25['Severe']:.0f}% of Severe days\n(against {pm25['Good']:.0f}% before Good days)",
            (5, pm25['Severe']), xytext=(2.3, 104), fontsize=9, arrowprops=dict(arrowstyle='->', color='grey'))
ax.annotate('Gases lead on clean days: when particles\nare low, a gas becomes the "worst" pollutant',
            (0, flag_share.loc['Good', 'has_co']), xytext=(-0.1, 88), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='grey'))
ax.set_xticks(range(len(AQI_CATEGORIES)), AQI_CATEGORIES)
ax.set_ylim(-3, 118)
ax.set_xlabel('tomorrow_status')
ax.set_ylabel('% of days flagged')
fig.suptitle('PM2.5 drives the bad days; the gases mark the clean ones  → keep the six flags',
             x=0.01, ha='left', fontsize=12, fontweight='bold')
ax.set_title("Prominent pollutant flags (today), by tomorrow's category", loc='left', fontsize=10)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.show()
flag_share.round(1)
```

![chart](eda_files/eda_46_5.png)

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>has_co</th>
      <th>has_no2</th>
      <th>has_o3</th>
      <th>has_pm10</th>
      <th>has_pm25</th>
      <th>has_so2</th>
    </tr>
    <tr>
      <th>tomorrow_status</th>
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
      <th>Good</th>
      <td>42.0</td>
      <td>21.8</td>
      <td>30.2</td>
      <td>63.9</td>
      <td>38.2</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>Satisfactory</th>
      <td>30.7</td>
      <td>9.2</td>
      <td>21.9</td>
      <td>66.0</td>
      <td>48.9</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>Moderate</th>
      <td>5.9</td>
      <td>3.3</td>
      <td>18.5</td>
      <td>58.5</td>
      <td>68.6</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>Poor</th>
      <td>0.9</td>
      <td>1.2</td>
      <td>8.9</td>
      <td>41.9</td>
      <td>80.9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Very Poor</th>
      <td>0.3</td>
      <td>0.4</td>
      <td>1.4</td>
      <td>27.8</td>
      <td>90.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Severe</th>
      <td>0.4</td>
      <td>0.2</td>
      <td>0.5</td>
      <td>21.8</td>
      <td>91.8</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

**Finding:** PM2.5 drives the bad days: it is prominent on about 92% of the days before a Severe day, against 38% before a Good day. CO, O<sub>3</sub> and NO<sub>2</sub> mostly appear before clean days: when particles are low, a gas becomes the "worst" pollutant even though the air is clean overall. PM10 is common on cleaner days (dust) and falls as AQI rises.
**Decision:** keep the six flags; they add information about *which* pollutant drives the AQI, which the AQI value alone does not show.

### 8.1 Mutual information: how much does each categorical feature tell us?

Correlation needs numbers. For categories (today's category, city, month, the pollutant flags), **mutual information** asks: *once I know this feature, how much less unsure am I about tomorrow's category?* **0** means it tells nothing; higher means more. There is no fixed top of the scale, so `vs_status_today` shows each score as a share of the strongest one.

Computed on the training years only (2015–2022), so the validation and test years stay unseen.

```python
from sklearn.metrics import mutual_info_score

train_part = df[df['date'].dt.year <= 2022].assign(month=df['date'].dt.month)   # training years only
mi_features = ['status_today', 'city', 'month'] + flag_cols

mi = pd.Series({c: mutual_info_score(train_part[target], train_part[c]) for c in mi_features})
mi = mi.sort_values(ascending=False).round(3).to_frame('mutual_info')
mi['vs_status_today'] = (mi['mutual_info'] / mi.loc['status_today', 'mutual_info']).round(2)
mi
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mutual_info</th>
      <th>vs_status_today</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>status_today</th>
      <td>0.686</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.227</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>city</th>
      <td>0.180</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>has_co</th>
      <td>0.105</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>has_pm25</th>
      <td>0.049</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>has_o3</th>
      <td>0.039</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>has_no2</th>
      <td>0.029</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>has_pm10</th>
      <td>0.028</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>has_so2</th>
      <td>0.009</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>

**Finding:** today's category tells by far the most about tomorrow (0.686), which is why "tomorrow = today" is a strong benchmark. Month (0.227) and city (0.180) come next, each carrying about a quarter to a third as much: this confirms the seasonality (section 6) and the city differences (section 3.1). Among the pollutant flags, CO (0.105) scores highest and SO<sub>2</sub> (0.009) almost nothing. CO marks clean days, which are many; PM2.5 marks Severe days, which are few, so its score is lower than its 92% share before Severe days suggests (mutual information is dominated by the common categories).
**Decision:** keep `city`, `month` and all six flags. The weak flags cost little, and tree models can ignore what does not help.

## 9. Collinearity

Collinear features carry the same information. This makes logistic regression coefficients unstable and hard to interpret; tree models are not affected. Two checks are used: the correlation matrix for pairs, and the VIF for each feature against **all** the others.

```python
corr = df.corr(numeric_only=True)
n = len(corr)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)      # hide the mirrored upper triangle

fig, ax = plt.subplots(figsize=(14, 12.5))
im = ax.imshow(corr.mask(mask), cmap='coolwarm', vmin=-1, vmax=1)
for i in range(n):
    for j in range(i + 1):
        v = corr.iloc[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(v) > 0.6 else 'black')
        if 0 < j < i and abs(v) >= 0.6:                   # outline strongly related feature pairs (not aqi_today)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1.5))
ax.set_xticks(range(n), corr.columns, rotation=45, ha='right')
ax.set_yticks(range(n), corr.columns)
ax.spines[:].set_visible(False)
fig.colorbar(im, shrink=0.8)
fig.suptitle('Some features move together (outlined: |r| ≥ 0.6): temperature, moisture, wind\n'
             '→ a reduced set for logistic regression (VIF check below); tree models keep all',
             x=0.01, ha='left', fontsize=13, fontweight='bold')
ax.set_title('Correlation matrix (Pearson). The aqi_today row shows relevance to AQI, not collinearity',
             loc='left', fontsize=10)
plt.tight_layout()
plt.show()
```

![chart](eda_files/eda_52_6.png)

```python
# Feature pairs with |r| >= 0.6 (the aqi_today row shows relevance, not collinearity)
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
(upper.stack().rename('r').to_frame()
      .assign(abs_r=lambda d: d['r'].abs())
      .query('abs_r >= 0.6')
      .sort_values('abs_r', ascending=False))
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>r</th>
      <th>abs_r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">temp_mean</th>
      <th>temp_min</th>
      <td>0.952840</td>
      <td>0.952840</td>
    </tr>
    <tr>
      <th>temp_max</th>
      <td>0.947986</td>
      <td>0.947986</td>
    </tr>
    <tr>
      <th>wind_speed_mean</th>
      <th>wind_speed_max</th>
      <td>0.880211</td>
      <td>0.880211</td>
    </tr>
    <tr>
      <th>temp_max</th>
      <th>temp_min</th>
      <td>0.816338</td>
      <td>0.816338</td>
    </tr>
    <tr>
      <th>dew_point_mean</th>
      <th>temp_range</th>
      <td>-0.777227</td>
      <td>0.777227</td>
    </tr>
    <tr>
      <th>temp_min</th>
      <th>dew_point_mean</th>
      <td>0.760958</td>
      <td>0.760958</td>
    </tr>
    <tr>
      <th>humidity_mean</th>
      <th>temp_range</th>
      <td>-0.740050</td>
      <td>0.740050</td>
    </tr>
    <tr>
      <th>wind_u_mean</th>
      <th>wind_dir_mean</th>
      <td>0.705834</td>
      <td>0.705834</td>
    </tr>
    <tr>
      <th>cloud_cover_mean</th>
      <th>temp_range</th>
      <td>-0.672468</td>
      <td>0.672468</td>
    </tr>
    <tr>
      <th>aqi_today</th>
      <th>dew_point_mean</th>
      <td>-0.605058</td>
      <td>0.605058</td>
    </tr>
  </tbody>
</table>
</div>

A correlation matrix only compares two features at a time. `temp_range` is exactly `temp_max − temp_min`, a three-way dependency that no single cell shows. The VIF (variance inflation factor) catches it. VIF is computed on the **training years only**: 1–5 is fine, 5–10 is moderate, above 10 is serious.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def vif_table(X):
    """VIF for each column of X. An intercept is added, as VIF requires.
    VIF needs complete rows, so rows with a missing or infinite value are skipped (and counted)."""
    complete = np.isfinite(X).all(axis=1)
    if not complete.all():
        print(f'VIF: skipped {(~complete).sum():,} rows with missing values. '
              'After section 5 there should be none: restart the kernel and run all cells.')
    Xc = add_constant(X[complete])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')        # perfectly collinear sets warn about a singular matrix
        vif = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    return pd.Series(vif, index=X.columns, name='VIF').sort_values(ascending=False).round(2)


df_train, df_val, df_test = split_train_val_test(df)

all_weather = ['aqi_today', 'temp_mean', 'temp_max', 'temp_min', 'temp_range', 'humidity_mean',
               'dew_point_mean', 'precip_sum', 'pressure_mean', 'cloud_cover_mean',
               'wind_speed_mean', 'wind_speed_max', 'wind_u_mean', 'wind_v_mean']
vif_table(df_train[all_weather]).to_frame('VIF: all weather features')
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF: all weather features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>temp_max</th>
      <td>1.000800e+15</td>
    </tr>
    <tr>
      <th>temp_min</th>
      <td>1.000800e+15</td>
    </tr>
    <tr>
      <th>temp_range</th>
      <td>1.000800e+15</td>
    </tr>
    <tr>
      <th>temp_mean</th>
      <td>2.282200e+02</td>
    </tr>
    <tr>
      <th>dew_point_mean</th>
      <td>4.571000e+01</td>
    </tr>
    <tr>
      <th>humidity_mean</th>
      <td>3.526000e+01</td>
    </tr>
    <tr>
      <th>wind_speed_mean</th>
      <td>5.550000e+00</td>
    </tr>
    <tr>
      <th>wind_speed_max</th>
      <td>5.460000e+00</td>
    </tr>
    <tr>
      <th>cloud_cover_mean</th>
      <td>2.410000e+00</td>
    </tr>
    <tr>
      <th>aqi_today</th>
      <td>2.330000e+00</td>
    </tr>
    <tr>
      <th>precip_sum</th>
      <td>1.400000e+00</td>
    </tr>
    <tr>
      <th>wind_v_mean</th>
      <td>1.360000e+00</td>
    </tr>
    <tr>
      <th>pressure_mean</th>
      <td>1.340000e+00</td>
    </tr>
    <tr>
      <th>wind_u_mean</th>
      <td>1.100000e+00</td>
    </tr>
  </tbody>
</table>
</div>

```python
lr_numerical = ['aqi_today', 'temp_min', 'temp_range', 'humidity_mean', 'precip_sum',
                'pressure_mean', 'cloud_cover_mean', 'wind_speed_mean',
                'wind_u_mean', 'wind_v_mean']
vif_table(df_train[lr_numerical]).to_frame('VIF: reduced set for logistic regression')
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF: reduced set for logistic regression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>temp_range</th>
      <td>7.38</td>
    </tr>
    <tr>
      <th>humidity_mean</th>
      <td>5.24</td>
    </tr>
    <tr>
      <th>temp_min</th>
      <td>3.25</td>
    </tr>
    <tr>
      <th>cloud_cover_mean</th>
      <td>2.36</td>
    </tr>
    <tr>
      <th>aqi_today</th>
      <td>2.33</td>
    </tr>
    <tr>
      <th>wind_speed_mean</th>
      <td>1.43</td>
    </tr>
    <tr>
      <th>wind_v_mean</th>
      <td>1.35</td>
    </tr>
    <tr>
      <th>pressure_mean</th>
      <td>1.33</td>
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

**Findings:**
- **Temperature.** `temp_max`, `temp_min` and `temp_range` are perfectly collinear (VIF ≈ 10<sup>15</sup>), because `temp_range` is exactly `temp_max − temp_min`. `temp_mean` is a near-duplicate as well (r = 0.95 with both `temp_min` and `temp_max`; VIF ≈ 228).
- **Moisture.** `dew_point_mean` (VIF ≈ 46) and `humidity_mean` (VIF ≈ 35) are also above 10, which is serious. Both fall as `temp_range` rises (r = −0.78 and −0.74): on humid days the gap between day and night temperature is small. `dew_point_mean` also follows `temp_min` (r = 0.76). `humidity_mean` does not (r ≈ 0); it falls as `temp_max` rises (r = −0.53).
- **Wind.** `wind_speed_mean` and `wind_speed_max` correlate at 0.88, but their VIF is only moderate (≈ 5.5 each). `wind_dir_mean` correlates 0.71 with `wind_u_mean`; it is left out of the VIF lists because it is an angle (359° and 1° are neighbours, but far apart as numbers).
- **Clouds.** `cloud_cover_mean` and `temp_range` correlate at −0.67 (cloudy days have smaller day–night swings), but the VIF of `cloud_cover_mean` is low (≈ 2.4), so no action is needed.
- **Reduced set.** Every VIF is below 10. Only `temp_range` (≈ 7.4) and `humidity_mean` (≈ 5.2) are above 5. Removing `humidity_mean` as well brings `temp_range` down to ≈ 2.8.

**Decisions:**
- For **logistic regression**, drop `temp_mean`, `temp_max`, `wind_speed_max`, `dew_point_mean` and `wind_dir_mean` (`wind_u_mean` and `wind_v_mean` encode the same direction without the 359°→0° jump). Test dropping `humidity_mean` as well on validation: it removes the last moderate VIF, and logistic regression cannot use its U-shape anyway (section 8).
- For **tree models**, keep all features.
- Repeat this check after adding the lag and rolling features: `aqi_roll7_mean` correlates 0.92 with `aqi_today` (section 10).

## 10. Why lag and rolling features?

**The problem.** The model looks at one row at a time. Each row is one city on one day, and the model does not read the rows in order. From `aqi_today` alone it cannot tell whether AQI has been climbing all week or jumped only today. A person forecasting tomorrow would look at the last few days, not only at today.

**The fix.** Copy a short summary of the recent past into today's row. Each feature answers one simple question:

| Feature | Question it answers | Delhi, 2 Nov 2023 |
|---|---|---|
| `aqi_roll7_mean` | What has the usual level been this week? | 336 |
| `aqi_roll7_std` | Has the week been calm or jumpy? | 43.5 |
| `aqi_diff1` | Is AQI rising or falling since yesterday? | +28 |
| `aqi_dev7` | How unusual is today compared with the previous 7 days? | +75 |
| `month` | What time of year is it? (section 6) | 11 |

**Three rules keep the features honest** (`add_trend_features` in `utility.py`):
1. **Only the past.** Every feature uses today and earlier days, never tomorrow, which is the answer.
2. **Within each city.** Delhi's past never ends up in Bengaluru's row.
3. **By calendar day.** "Yesterday" is always the previous date. If that day is missing, the feature is left empty (NaN) instead of quietly using an older day.

### 10.1 Does AQI remember its past?

If today's AQI had nothing to do with last week's, the past would be useless to the model. **Autocorrelation** measures this: the correlation between a city's AQI today and its AQI *k* days earlier. How fast it fades also tells us how long the window should be.

```python
# Autocorrelation: how similar is AQI to its value k days earlier? (per city, then averaged)
lags = range(1, 15)
acf = pd.DataFrame({
    city: [s.autocorr(k) for k in lags]
    for city, s in ((c, g.set_index('date')['aqi_today'].asfreq('D')) for c, g in df.groupby('city'))
}, index=lags)
avg = acf.mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(acf.index, avg, color=COLOR, linewidth=2, marker='o', label='average of 14 cities')
ax.fill_between(acf.index, acf.min(axis=1), acf.max(axis=1), color=COLOR, alpha=0.15, label='range across cities')
ax.axvline(7, color='grey', linestyle='--', linewidth=1)
ax.text(7.15, 0.9, '7-day window used for\nthe rolling features', fontsize=9, color='#555555')
ax.annotate(f'Yesterday: r = {avg[1]:.2f}', (1, avg[1]), xytext=(1.6, 0.9), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='grey'))
ax.annotate(f'The drop slows after about 5 days:\nonly the season remains (r ≈ {avg[14]:.2f} at 14 days)',
            (14, avg[14]), xytext=(9.3, 0.45), fontsize=9, arrowprops=dict(arrowstyle='->', color='grey'))
ax.set_xticks(list(lags))
ax.set_xlabel('Days back (lag)')
ax.set_ylabel('Correlation with today')
fig.suptitle('AQI has a strong memory, mostly of the last few days  → add 7-day lag and rolling features',
             x=0.01, ha='left', fontsize=12, fontweight='bold')
ax.set_title("Correlation of each city's AQI with its own value k days earlier", loc='left', fontsize=10)
ax.legend(frameon=False, loc='lower left')
plt.tight_layout()
plt.show()
acf.mean(axis=1).round(2).to_frame('avg correlation').T
```

![chart](eda_files/eda_59_7.png)

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>avg correlation</th>
      <td>0.86</td>
      <td>0.78</td>
      <td>0.73</td>
      <td>0.7</td>
      <td>0.68</td>
      <td>0.67</td>
      <td>0.66</td>
      <td>0.65</td>
      <td>0.64</td>
      <td>0.64</td>
      <td>0.63</td>
      <td>0.63</td>
      <td>0.62</td>
      <td>0.62</td>
    </tr>
  </tbody>
</table>
</div>

**Finding:** AQI has a strong memory. Today is closely tied to yesterday (r ≈ 0.86 on average). The link drops quickly over the first five days (0.86 → 0.68) and then hardly moves (0.66 at 7 days, 0.62 at 14). That flat tail is the season: winter stays bad for weeks, and `month` already captures that. The cleanest cities, Chennai and Bengaluru, forget fastest.

**Decision:** use a **7-day window**. It covers the part of the memory that changes from day to day; a longer window would mostly repeat the season.

### 10.2 Building the features

```python
trend = add_trend_features(df)      # in modelling, call this before dropping rows (see the docstring)
trend_cols = ['aqi_roll7_mean', 'aqi_roll7_std', 'aqi_diff1', 'aqi_dev7']

# One week in Delhi, read row by row
(trend.loc[(trend['city'] == 'Delhi') & trend['date'].between('2023-10-27', '2023-11-02'),
           ['date', 'aqi_today'] + trend_cols + ['month', 'tomorrow_status']]
      .set_index('date').round(1))
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
      <th>month</th>
      <th>tomorrow_status</th>
    </tr>
    <tr>
      <th>date</th>
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
      <th>2023-10-27</th>
      <td>261.0</td>
      <td>257.7</td>
      <td>28.4</td>
      <td>5.0</td>
      <td>12.7</td>
      <td>10</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>2023-10-28</th>
      <td>304.0</td>
      <td>265.7</td>
      <td>32.7</td>
      <td>43.0</td>
      <td>46.3</td>
      <td>10</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>2023-10-29</th>
      <td>325.0</td>
      <td>267.4</td>
      <td>35.8</td>
      <td>21.0</td>
      <td>59.3</td>
      <td>10</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>2023-10-30</th>
      <td>347.0</td>
      <td>279.4</td>
      <td>46.5</td>
      <td>22.0</td>
      <td>79.6</td>
      <td>10</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>2023-10-31</th>
      <td>359.0</td>
      <td>299.3</td>
      <td>46.6</td>
      <td>12.0</td>
      <td>79.6</td>
      <td>10</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>2023-11-01</th>
      <td>364.0</td>
      <td>316.6</td>
      <td>44.6</td>
      <td>5.0</td>
      <td>64.7</td>
      <td>11</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>2023-11-02</th>
      <td>392.0</td>
      <td>336.0</td>
      <td>43.5</td>
      <td>28.0</td>
      <td>75.4</td>
      <td>11</td>
      <td>Severe</td>
    </tr>
  </tbody>
</table>
</div>

Reading the Delhi week: AQI rose every day. `aqi_roll7_mean` follows behind today's value (336 against 392), because it averages the whole week. `aqi_diff1` shows the daily rises, and `aqi_dev7` says today is 75 points above the previous week. The next day was *Severe*. `aqi_today` alone (392) shows a high level but not that it got there by climbing for a week.

### 10.3 Do the new features carry signal?

The same check as section 8: how each feature moves across tomorrow's classes, and how much it overlaps with `aqi_today`.

```python
class_signal_summary(trend.dropna(subset=trend_cols), ['aqi_today'] + trend_cols)
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
      <th>avg_box_overlap</th>
      <th>good_vs_severe_overlap</th>
      <th>linear_r2</th>
      <th>shape</th>
      <th>same_as</th>
    </tr>
    <tr>
      <th>feature</th>
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
      <th>aqi_today</th>
      <td>0.88</td>
      <td>up</td>
      <td>46 → 73 → 139 → 236 → 332 → 407</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.98</td>
      <td>straight</td>
      <td>-</td>
    </tr>
    <tr>
      <th>aqi_roll7_mean</th>
      <td>0.85</td>
      <td>up</td>
      <td>52.1 → 77.9 → 139.7 → 229.1 → 314.4 → 363.9</td>
      <td>0.07</td>
      <td>0.00</td>
      <td>0.98</td>
      <td>straight</td>
      <td>-</td>
    </tr>
    <tr>
      <th>aqi_roll7_std</th>
      <td>0.61</td>
      <td>up</td>
      <td>9.6 → 14.4 → 31.1 → 43.6 → 46.5 → 49</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>0.92</td>
      <td>bends</td>
      <td>-</td>
    </tr>
    <tr>
      <th>aqi_dev7</th>
      <td>0.22</td>
      <td>weak up</td>
      <td>-6.4 → -4 → 1.7 → 9.1 → 22.2 → 49.4</td>
      <td>0.89</td>
      <td>0.00</td>
      <td>0.86</td>
      <td>bends</td>
      <td>Good≈Satisfactory</td>
    </tr>
    <tr>
      <th>aqi_diff1</th>
      <td>0.14</td>
      <td>weak up</td>
      <td>-2 → -1 → 2 → 6 → 9 → 21.5</td>
      <td>0.95</td>
      <td>0.86</td>
      <td>0.87</td>
      <td>bends</td>
      <td>Good≈Satisfactory</td>
    </tr>
  </tbody>
</table>
</div>

```python
trend[['aqi_today'] + trend_cols].corr().round(2)
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
      <td>0.92</td>
      <td>0.53</td>
      <td>0.21</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>aqi_roll7_mean</th>
      <td>0.92</td>
      <td>1.00</td>
      <td>0.59</td>
      <td>-0.05</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <th>aqi_roll7_std</th>
      <td>0.53</td>
      <td>0.59</td>
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
      <td>0.37</td>
      <td>-0.03</td>
      <td>-0.05</td>
      <td>0.60</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>

**Findings:**
- `aqi_roll7_mean` is strong (rho = 0.85), but it correlates 0.92 with `aqi_today`. It is mostly the same information, smoothed.
- `aqi_roll7_std` is the most useful new feature: a clear signal (rho = 0.61) that is only partly tied to `aqi_today` (r = 0.53), so it adds information. Jumpy weeks come before bad days.
- `aqi_diff1` and `aqi_dev7` are weak on their own (rho = 0.14 and 0.22) and almost unrelated to the weekly level. They may still help when combined with weather, which tree models can find.

### 10.4 The hard case: Very Poor today, Severe tomorrow?

Section 8 showed that weather cannot separate Very Poor from Severe. Can the trend? Take the days that are Very Poor today and compare how often tomorrow turns Severe.

```python
vp = trend[trend['status_today'] == 'Very Poor'].dropna(subset=trend_cols)
is_severe = vp['tomorrow_status'].eq('Severe')

pd.concat({
    'AQI level today': is_severe.groupby(pd.cut(vp['aqi_today'], [300, 340, 370, 400])).mean(),
    'Rising or falling (aqi_diff1)': is_severe.groupby(np.where(vp['aqi_diff1'] > 0, 'rising', 'falling or flat')).mean(),
    'Jump vs last week (aqi_dev7)': is_severe.groupby(pd.qcut(vp['aqi_dev7'], 4, labels=['lowest 25%', '2nd', '3rd', 'highest 25%'])).mean(),
}, names=['split by', 'group']).mul(100).round(1).to_frame('% Severe tomorrow')
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>% Severe tomorrow</th>
    </tr>
    <tr>
      <th>split by</th>
      <th>group</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">AQI level today</th>
      <th>(300, 340]</th>
      <td>3.1</td>
    </tr>
    <tr>
      <th>(340, 370]</th>
      <td>10.8</td>
    </tr>
    <tr>
      <th>(370, 400]</th>
      <td>23.8</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Rising or falling (aqi_diff1)</th>
      <th>falling or flat</th>
      <td>9.8</td>
    </tr>
    <tr>
      <th>rising</th>
      <td>9.6</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Jump vs last week (aqi_dev7)</th>
      <th>lowest 25%</th>
      <td>10.3</td>
    </tr>
    <tr>
      <th>2nd</th>
      <td>10.7</td>
    </tr>
    <tr>
      <th>3rd</th>
      <td>9.9</td>
    </tr>
    <tr>
      <th>highest 25%</th>
      <td>7.9</td>
    </tr>
  </tbody>
</table>
</div>

**Finding:** the **level** matters most. A Very Poor day at 370–400 turns Severe 23.8% of the time, against 3.1% at 300–340. The **direction** does not help: about 10% whether AQI is rising or falling. The biggest jumps are even slightly *less* likely to turn Severe (7.9%), because AQI tends to fall back after a spike.

**Decisions:**
- Add the four features and `month`. They are cheap, use only the past, and give the model the recent history it cannot see otherwise.
- `aqi_roll7_std` is the most promising new feature. Keep `aqi_diff1` and `aqi_dev7` only if they improve the validation score.
- For **logistic regression**, `aqi_roll7_mean` nearly duplicates `aqi_today` (r = 0.92): use one of them and rerun the VIF check. Tree models can take both.
- Telling Very Poor from Severe stays the hardest part. The clearest clue is how high AQI already is, not where it is heading.

## 11. Summary of decisions

| Topic | Finding | Decision |
|---|---|---|
| Labels | 56 labels disagree with their AQI value | Build the target and `status_today` from the AQI value (`aqi_to_category`) |
| Target | Tomorrow's category, via `shift(-1)` per city | Built **before** filling gaps; never filled |
| Missing AQI | 4.2% of days; 90% of gaps ≤ 3 days | Forward fill ≤ 3 days per city (past only), flag with `aqi_imputed`, drop the rest |
| Stations | No relationship with AQI | Drop `number_of_monitoring_stations` |
| `city` | Strong differences in baseline AQI | One-hot encode; test with and without on validation |
| Pollutants | 52 strings from 7 pollutants | Six `has_*` flags; drop NH<sub>3</sub> |
| Categorical signal | Mutual information: today's category 0.69, month 0.23, city 0.18, pollutant flags 0.01–0.11 | Keep all; today's category carries most of the signal |
| Class balance | Severe 2.4%, Satisfactory 32.8% | Macro F1, Very Poor and Severe recall, `class_weight='balanced'` |
| Seasonality | Winter is worst; Severe peaks in November | Add `month` / season features |
| Skew | `precip_sum` skew ≈ 7.6 | `log1p` or a `rained` flag for logistic regression |
| Pressure | Peaks reflect city altitude | Use pressure relative to the city's training mean, or drop |
| Outliers | All plausible, real events | Keep |
| Signal | `aqi_today` strongest; weather separates clean from polluted, not Very Poor from Severe | Add lag and rolling features (section 10) |
| Memory | AQI tracks its recent past (r ≈ 0.86 with yesterday); the link fades over about 5 days, then only the season remains | 7-day per-city features: `aqi_roll7_mean`, `aqi_roll7_std`, `aqi_diff1`, `aqi_dev7` |
| Very Poor vs Severe | Depends on how high AQI already is, not on whether it is rising | Keep `aqi_diff1` / `aqi_dev7` only if they help on validation |
| Shape | Weather patterns bend; humidity reverses | Expect tree models to beat logistic regression |
| Collinearity | Temperature trio, wind pair, dew point | Reduced feature set for logistic regression; all features for trees |

**Next steps**
1. Build the lag and rolling features and `month` with `add_trend_features`, after `ffill_aqi` and **before** dropping rows (a dropped day's AQI is still a valid past value). Rerun the VIF check with them.
2. Split with `split_train_val_test` (train 2015–2022, validation 2023, test 2024–2025).
3. Score the persistence baseline (tomorrow = today), then logistic regression, random forest and XGBoost on validation; evaluate the best model once on test.

**Later updates.** The table above records the EDA decisions as they were taken. Modelling changed three of them (details in [`docs/model_decisions.md`](../docs/model_decisions.md)):
- **Class weights:** tested for every model rather than assumed. The final XGBoost model uses none (Decision 3).
- **Split:** the final model is trained on 2015–2024 and tested once on 2025 (Decision 11).
- **Models:** logistic regression, a decision tree, a random forest and XGBoost were all scored. As section 8 expected, the tree models won (Decision 12).
