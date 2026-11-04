<!-- Exported from experiments/weather_window.ipynb: every cell run top to bottom; generated, do not edit by hand -->

# Weather on the Bulletin's Window: Calendar Day or 4 PM to 4 PM?

**The question.** The CPCB bulletin published at 4 PM on day D averages the 24 hours **from 4 PM on D-1 to 4 PM on D**. The model's weather inputs cover the **calendar day** D (midnight to midnight), so their last 8 hours (4 PM to midnight on D) belong to the **next** bulletin's window, the one being forecast.

| | Hours covered |
|---|---|
| Today's AQI (bulletin of D) | 4 PM on D-1 to 4 PM on D |
| Today's weather (current inputs) | midnight to midnight on D |
| Tomorrow's AQI (the target, bulletin of D+1) | 4 PM on D to 4 PM on D+1 |

**The experiment.** The same Open-Meteo hourly data is aggregated over each bulletin's own window (`code/download_weather_window.py`), and the final model's settings are trained with each weather version:
- **calendar day**: the current inputs;
- **bulletin window**: weather of 4 PM on D-1 to 4 PM on D, all measured by the time the bulletin is out;
- **bulletin window + target window** (ceiling, not deployable): also the measured weather of the target's window.

**Scope:** the five NCR cities whose hourly data was downloaded (Delhi, Noida, Greater Noida, Ghaziabad, Gurugram), where most errors are. Every version is trained and scored on exactly the same city-days, with the usual rules: rolling validation 2019-2023, the 2025 test, and the October 2026 live days.

## 1. Setup and data


```python
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, '../scripts')
sys.path.insert(0, '../code')
import train as T
from utility import AQI_CATEGORIES, CategoryXGB, aqi_to_category
from download_weather import to_daily

warnings.filterwarnings('ignore', category=UserWarning)
CITIES = ['Delhi', 'Noida', 'Greater Noida', 'Ghaziabad', 'Gurugram']
WEATHER = ['temp_mean', 'temp_max', 'temp_min', 'temp_range', 'humidity_mean', 'dew_point_mean', 'precip_sum',
           'pressure_mean', 'cloud_cover_mean', 'wind_speed_mean', 'wind_speed_max', 'wind_u_mean', 'wind_v_mean']

df = T.load_data(Path('../data/processed/aqi_weather_2015_2025.csv'))
df = df[df['city'].isin(CITIES)].reset_index(drop=True)
window = pd.read_csv('../data/processed/weather_bulletin_window.csv', parse_dates=['date'])
aligned = df.drop(columns=WEATHER).merge(window[['city', 'date'] + WEATHER], on=['city', 'date'], how='left')

target = window[['city', 'date'] + WEATHER].copy()
target['date'] -= pd.Timedelta(days=1)                       # the next bulletin's window, on today's row
target = target.rename(columns={w: f'target_{w}' for w in WEATHER})
ceiling = aligned.merge(target, on=['city', 'date'], how='left')
print(f'{len(df):,} city-days; days without bulletin-window weather: {aligned[WEATHER].isna().any(axis=1).sum()}')
```

    16,195 city-days; days without bulletin-window weather: 0


**Check:** the new hourly download, aggregated per calendar day, must reproduce the dataset's weather, so that the only difference between the versions is the window.


```python
hourly = Path('../data/raw/weather_hourly/delhi.csv')
if hourly.exists():                                          # the hourly cache is not in git
    calendar = to_daily(pd.read_csv(hourly))
    both = df[df['city'] == 'Delhi'][['date'] + WEATHER].merge(calendar[['date'] + WEATHER], on='date', suffixes=('', '_new'))
    print(f'Delhi, {len(both):,} days: largest difference {max((both[w] - both[w + "_new"]).abs().max() for w in WEATHER):.3f}')
else:
    print('Run `python code/download_weather_window.py` to download the hourly data for this check.')
```

    Delhi, 3,866 days: largest difference 0.000


## 2. Rolling validation, 2019-2023


```python
VERSIONS = {'calendar day (current)': (df, T.FEATURES),
            'bulletin window': (aligned, T.FEATURES),
            'bulletin window + target window (ceiling)': (ceiling, T.FEATURES + [f'target_{w}' for w in WEATHER])}

def fit(part, features):
    model = Pipeline([('dv', DictVectorizer(sparse=False)), ('m', CategoryXGB(**T.PARAMS))])
    return model.fit(part[features].to_dict('records'), part[T.TARGET].astype(str).to_numpy())

def scores(model, part, features):
    y = part[T.TARGET].astype(str).to_numpy()
    pred = model.predict(part[features].to_dict('records'))
    changed = part['status_today'].astype(str).to_numpy() != y
    return {'accuracy': (pred == y).mean(), 'macro F1': f1_score(y, pred, labels=AQI_CATEGORIES, average='macro'),
            'on change days': (pred[changed] == y[changed]).mean(), 'on steady days': (pred[~changed] == y[~changed]).mean()}

rows = []
for name, (data, features) in VERSIONS.items():
    for year in range(2019, 2024):
        model = fit(data[data['date'].dt.year < year], features)
        rows.append({'version': name, 'year': year, **scores(model, data[data['date'].dt.year == year], features)})
rolling = pd.DataFrame(rows)
rolling.pivot(index='year', columns='version', values='accuracy')[list(VERSIONS)].round(3)
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
      <th>version</th>
      <th>calendar day (current)</th>
      <th>bulletin window</th>
      <th>bulletin window + target window (ceiling)</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019</th>
      <td>0.580</td>
      <td>0.577</td>
      <td>0.623</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>0.675</td>
      <td>0.661</td>
      <td>0.664</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>0.673</td>
      <td>0.659</td>
      <td>0.691</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>0.668</td>
      <td>0.639</td>
      <td>0.683</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>0.705</td>
      <td>0.690</td>
      <td>0.700</td>
    </tr>
  </tbody>
</table>
</div>




```python
rolling.groupby('version', sort=False)[['accuracy', 'macro F1', 'on change days', 'on steady days']].mean().round(3)
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
      <th>macro F1</th>
      <th>on change days</th>
      <th>on steady days</th>
    </tr>
    <tr>
      <th>version</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>calendar day (current)</th>
      <td>0.660</td>
      <td>0.573</td>
      <td>0.378</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>bulletin window</th>
      <td>0.645</td>
      <td>0.538</td>
      <td>0.339</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>bulletin window + target window (ceiling)</th>
      <td>0.672</td>
      <td>0.590</td>
      <td>0.410</td>
      <td>0.83</td>
    </tr>
  </tbody>
</table>
</div>



## 3. The 2025 test and the October 2026 live days


```python
live = pd.read_csv('../data/processed/live_check_2026_10.csv', parse_dates=['date']).dropna(subset=['tomorrow_aqi'])
live = live[live['city'].isin(CITIES)].reset_index(drop=True)
live[T.TARGET] = aqi_to_category(live['tomorrow_aqi']).astype(str)
live['status_today'] = aqi_to_category(live['aqi_today']).astype(str)
live_window = pd.read_csv('../data/processed/weather_bulletin_window_live_check.csv', parse_dates=['date'])
live_aligned = live.drop(columns=WEATHER).merge(live_window[['city', 'date'] + WEATHER], on=['city', 'date'], how='left')

rows = [{'version': 'persistence', 'test 2025': (df[df['date'].dt.year == 2025]['status_today'].astype(str) == df[df['date'].dt.year == 2025][T.TARGET].astype(str)).mean(),
         'live Oct 2026': (live['status_today'] == live[T.TARGET]).mean()}]
for name, (data, live_part) in [('calendar day (current)', (df, live)), ('bulletin window', (aligned, live_aligned))]:
    model = fit(data[data['date'].dt.year <= 2024], T.FEATURES)
    rows.append({'version': name,
                 'test 2025': scores(model, data[data['date'].dt.year == 2025], T.FEATURES)['accuracy'],
                 'live Oct 2026': scores(model, live_part, T.FEATURES)['accuracy']})
pd.DataFrame(rows).set_index('version').round(3)
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
      <th>test 2025</th>
      <th>live Oct 2026</th>
    </tr>
    <tr>
      <th>version</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>persistence</th>
      <td>0.651</td>
      <td>0.555</td>
    </tr>
    <tr>
      <th>calendar day (current)</th>
      <td>0.677</td>
      <td>0.619</td>
    </tr>
    <tr>
      <th>bulletin window</th>
      <td>0.683</td>
      <td>0.613</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Conclusion

- **The 8 evening hours carry real information about tomorrow.** Moving the weather to the bulletin's window, which drops them, costs about **1.5 points** of accuracy in validation, in every one of the five years, mostly on days the category changes. Those hours are the start of the target's window: in effect a very short-range forecast.
- **This is not a leak in practice.** In training those hours are measured; live, at about 5 PM, the service gets them from Open-Meteo as short-range estimates (0-8 hours ahead), so it keeps most of the benefit. **The current calendar-day weather stays.**
- The 2025 test and the live days move by about half a point either way: within noise next to five years of validation.
- **Weather for the whole target window would add about 1 point** over the current inputs (the ceiling). That is the only weather gain left, and it needs an archive of past weather forecasts to train on honestly.
