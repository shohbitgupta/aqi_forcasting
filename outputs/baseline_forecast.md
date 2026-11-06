<!-- Exported from experiments/baseline_forecast.ipynb: every cell run top to bottom; generated, do not edit by hand -->

# Persistence Baseline: How Good Is "Tomorrow = Today"?

A score such as "71% accurate" is only meaningful relative to what is achievable without a model. This notebook scores four rule-based forecasters (no ML) on validation and keeps the strongest as the **baseline** every model must beat.

Metrics are computed directly with pandas (counts and ratios); section 10 cross-checks them against scikit-learn.

## 1. Setup

```python
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '../scripts')                 # utility.py lives in scripts/
from utility import AQI_CATEGORIES, aqi_to_category, prepare_dataset, split_train_val_test, evaluate

DATA_PATH = '../data/processed/aqi_weather_2015_2025.csv'
TARGET = 'tomorrow_status'
TREND = ['aqi_roll7_mean', 'aqi_roll7_std', 'aqi_diff1', 'aqi_dev7']
```

## 2. The same days the models will use

Evaluation conditions are identical to the models':

| Rule | How |
|---|---|
| **Same days** | Drop the same 195 rows (empty trend features) that the models drop, in every split |
| **No look-ahead** | A forecaster may only use what is known **today** |
| **Test set stays locked** | All comparisons happen on **validation (2023)**. The test years are opened once, at the very end |

```python
df = prepare_dataset(DATA_PATH)
df = df[df[TREND].notna().all(axis=1)]          # the same 195 rows the models drop
df_train, df_val, df_test = split_train_val_test(df)

print('train:', len(df_train), '| validation:', len(df_val), '| test:', len(df_test))
```

**Output:**

```
train: 33543 | validation: 5104 | test: 9985
```

## 3. Four rule-based forecasters

Each uses exactly **one** source of information:

| Forecaster | Rule | Information it uses |
|---|---|---|
| **A** | Always say the most common category | Nothing about the day (the **prior**) |
| **B** | Say the most common category for this city in this month | **Where and when** (called *climatology*) |
| **C, persistence** | Say tomorrow will be the same as today | **Today** |
| **D** | Say the category of the average AQI over the last 7 days | **The recent week** |

A and B are fitted on the training years (2015–2022) only; C and D use only the city's own recent days. Below: forecaster B's prediction for Delhi, per month.

```python
# Forecaster A: always say the most common category of the training years
most_common = df_train[TARGET].value_counts().idxmax()
print('Most common category in 2015-2022:', most_common)
```

**Output:**

```
Most common category in 2015-2022: Satisfactory
```

```python
# Forecaster B: say the most common category for that city in that month (learned from training years)
typical = (df_train.groupby(['city', 'month'])[TARGET]
                   .agg(lambda s: s.value_counts().idxmax())
                   .rename('typical'))
typical.loc['Delhi']
```

**Output:**

```
month
1        Very Poor
2             Poor
3             Poor
4             Poor
5             Poor
6         Moderate
7     Satisfactory
8     Satisfactory
9         Moderate
10            Poor
11       Very Poor
12       Very Poor
Name: typical, dtype: category
Categories (6, str): ['Good' < 'Satisfactory' < 'Moderate' < 'Poor' < 'Very Poor' < 'Severe']
```

Forecaster D maps `aqi_roll7_mean` (mean AQI of today and the previous 6 days) to its CPCB category with `aqi_to_category` (0–50 Good, 51–100 Satisfactory, …, 401–500 Severe). It tests whether a smoothed weekly view predicts better than today alone.

```python
# Forecaster D: the category of the last 7 days' average AQI (today included)
week_average = aqi_to_category(df_val['aqi_roll7_mean']).astype(str)

df_val[['city', 'date', 'aqi_today', 'aqi_roll7_mean']].assign(week_average=week_average).head()
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
      <th>aqi_roll7_mean</th>
      <th>week_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bengaluru</td>
      <td>2023-01-01</td>
      <td>93.0</td>
      <td>60.571429</td>
      <td>Satisfactory</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bengaluru</td>
      <td>2023-01-02</td>
      <td>86.0</td>
      <td>67.142857</td>
      <td>Satisfactory</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bengaluru</td>
      <td>2023-01-03</td>
      <td>82.0</td>
      <td>72.857143</td>
      <td>Satisfactory</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bengaluru</td>
      <td>2023-01-04</td>
      <td>90.0</td>
      <td>79.000000</td>
      <td>Satisfactory</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bengaluru</td>
      <td>2023-01-05</td>
      <td>87.0</td>
      <td>83.714286</td>
      <td>Satisfactory</td>
    </tr>
  </tbody>
</table>
</div>

Actual category and the four forecasts, one row per validation day:

```python
# Put the real answer and the four forecasts side by side, one row per validation day
val = pd.DataFrame({
    'city': df_val['city'],
    'date': df_val['date'],
    'actual': df_val[TARGET].astype(str),                  # what really happened tomorrow
    'always_common': most_common,                          # forecaster A
    'city_month': [typical[(c, m)] for c, m in zip(df_val['city'], df_val['month'])],   # forecaster B
    'persistence': df_val['status_today'].astype(str),     # forecaster C: tomorrow = today
    'week_average': week_average,                          # forecaster D: category of the 7-day average
})
val[val['city'] == 'Delhi'].iloc[300:310]
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>date</th>
      <th>actual</th>
      <th>always_common</th>
      <th>city_month</th>
      <th>persistence</th>
      <th>week_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1028</th>
      <td>Delhi</td>
      <td>2023-10-28</td>
      <td>Very Poor</td>
      <td>Satisfactory</td>
      <td>Poor</td>
      <td>Very Poor</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>1029</th>
      <td>Delhi</td>
      <td>2023-10-29</td>
      <td>Very Poor</td>
      <td>Satisfactory</td>
      <td>Poor</td>
      <td>Very Poor</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>1030</th>
      <td>Delhi</td>
      <td>2023-10-30</td>
      <td>Very Poor</td>
      <td>Satisfactory</td>
      <td>Poor</td>
      <td>Very Poor</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>1031</th>
      <td>Delhi</td>
      <td>2023-10-31</td>
      <td>Very Poor</td>
      <td>Satisfactory</td>
      <td>Poor</td>
      <td>Very Poor</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>Delhi</td>
      <td>2023-11-01</td>
      <td>Very Poor</td>
      <td>Satisfactory</td>
      <td>Very Poor</td>
      <td>Very Poor</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>Delhi</td>
      <td>2023-11-02</td>
      <td>Severe</td>
      <td>Satisfactory</td>
      <td>Very Poor</td>
      <td>Very Poor</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>Delhi</td>
      <td>2023-11-03</td>
      <td>Severe</td>
      <td>Satisfactory</td>
      <td>Very Poor</td>
      <td>Severe</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>Delhi</td>
      <td>2023-11-04</td>
      <td>Severe</td>
      <td>Satisfactory</td>
      <td>Very Poor</td>
      <td>Severe</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>Delhi</td>
      <td>2023-11-05</td>
      <td>Severe</td>
      <td>Satisfactory</td>
      <td>Very Poor</td>
      <td>Severe</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <th>1037</th>
      <td>Delhi</td>
      <td>2023-11-06</td>
      <td>Very Poor</td>
      <td>Satisfactory</td>
      <td>Very Poor</td>
      <td>Severe</td>
      <td>Severe</td>
    </tr>
  </tbody>
</table>
</div>

**Delhi, early November 2023:** on 2 November the category jumped from Very Poor to Severe; persistence said Very Poor. On 6 November it improved to Very Poor; persistence still said Severe. **Persistence lags every transition by one day.**

The 7-day average (D) lags further: the air was Severe from 2 November, but the average only reached Severe on **6 November**, because it still includes the calmer days of the previous week. Reacting late is the weakness a model has to fix.

## 4. Accuracy

Accuracy = **correct days ÷ all days**.

```python
correct = val['persistence'] == val['actual']        # True where the forecast was right
print('Days:', len(val))
print('Correct:', correct.sum())
print('Accuracy:', round(correct.sum() / len(val), 3))
```

**Output:**

```
Days: 5104
Correct: 3626
Accuracy: 0.71
```

## 5. Confusion table

Days counted per (actual, forecast) pair with `pd.crosstab`. **Rows** = actual category tomorrow, **columns** = persistence forecast, **diagonal** = correct days.

```python
def confusion_table(actual, predicted):
    """Count the days for every (actual, predicted) pair. Rows = actual, columns = predicted."""
    table = pd.crosstab(actual, predicted)
    return table.reindex(index=AQI_CATEGORIES, columns=AQI_CATEGORIES, fill_value=0)

cm = confusion_table(val['actual'], val['persistence'])
cm
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>persistence</th>
      <th>Good</th>
      <th>Satisfactory</th>
      <th>Moderate</th>
      <th>Poor</th>
      <th>Very Poor</th>
      <th>Severe</th>
    </tr>
    <tr>
      <th>actual</th>
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
      <td>183</td>
      <td>108</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Satisfactory</th>
      <td>105</td>
      <td>1403</td>
      <td>268</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Moderate</th>
      <td>5</td>
      <td>273</td>
      <td>1310</td>
      <td>227</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poor</th>
      <td>0</td>
      <td>2</td>
      <td>230</td>
      <td>457</td>
      <td>85</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Very Poor</th>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>87</td>
      <td>246</td>
      <td>20</td>
    </tr>
    <tr>
      <th>Severe</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>21</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>

## 6. Recall, precision and F1 for one category

Computed for Severe from the confusion table:

| Score | Definition | From the table |
|---|---|---|
| **Recall** | Share of real Severe days forecast as Severe | diagonal ÷ **row** total |
| **Precision** | Share of "Severe" forecasts that were correct | diagonal ÷ **column** total |
| **F1** | High only when **both** are high | 2 × precision × recall ÷ (precision + recall) |

Low recall means missed Severe days; low precision means false alarms.

```python
c = 'Severe'
caught = cm.loc[c, c]                    # Severe days that persistence called Severe
really = cm.loc[c].sum()                 # all days that really were Severe (the row)
called = cm[c].sum()                     # all days persistence called Severe (the column)

recall = caught / really
precision = caught / called
f1 = 2 * precision * recall / (precision + recall)
print(f'Severe: caught {caught} of {really} real Severe days -> recall {recall:.2f}')
print(f'Severe: {caught} of {called} "Severe" forecasts were right -> precision {precision:.2f}')
print(f'Severe: F1 = {f1:.2f}')
```

**Output:**

```
Severe: caught 27 of 51 real Severe days -> recall 0.53
Severe: 27 of 52 "Severe" forecasts were right -> precision 0.52
Severe: F1 = 0.52
```

## 7. Every category, and macro F1

Same computation per category. Precision is set to 0 for a category that is never predicted (avoids division by zero).

**Macro F1** = plain average of the six F1 scores, so rare Severe days weigh as much as common Satisfactory days.

```python
def class_scores(actual, predicted):
    """Recall, precision and F1 for every category, calculated from the confusion table."""
    cm = confusion_table(actual, predicted)
    rows = []
    for c in AQI_CATEGORIES:
        caught = cm.loc[c, c]
        really = cm.loc[c].sum()
        called = cm[c].sum()
        recall = caught / really if really > 0 else 0.0
        precision = caught / called if called > 0 else 0.0          # never predicted -> 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append({'category': c, 'real days': really, 'caught': caught, 'predicted': called,
                     'recall': recall, 'precision': precision, 'f1': f1})
    return pd.DataFrame(rows).set_index('category')

scores = class_scores(val['actual'], val['persistence'])
scores.round(2)
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>real days</th>
      <th>caught</th>
      <th>predicted</th>
      <th>recall</th>
      <th>precision</th>
      <th>f1</th>
    </tr>
    <tr>
      <th>category</th>
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
      <td>293</td>
      <td>183</td>
      <td>293</td>
      <td>0.62</td>
      <td>0.62</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>Satisfactory</th>
      <td>1785</td>
      <td>1403</td>
      <td>1786</td>
      <td>0.79</td>
      <td>0.79</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>Moderate</th>
      <td>1829</td>
      <td>1310</td>
      <td>1826</td>
      <td>0.72</td>
      <td>0.72</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>Poor</th>
      <td>778</td>
      <td>457</td>
      <td>782</td>
      <td>0.59</td>
      <td>0.58</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>Very Poor</th>
      <td>368</td>
      <td>246</td>
      <td>365</td>
      <td>0.67</td>
      <td>0.67</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>Severe</th>
      <td>51</td>
      <td>27</td>
      <td>52</td>
      <td>0.53</td>
      <td>0.52</td>
      <td>0.52</td>
    </tr>
  </tbody>
</table>
</div>

```python
macro_f1 = scores['f1'].mean()            # every category gets an equal vote
print('Macro F1 (persistence):', round(macro_f1, 3))
```

**Output:**

```
Macro F1 (persistence): 0.651
```

## 8. Comparing the four forecasters

Same scorecard for all four, ordered by how recent the information each one uses.

```python
def scorecard(name, actual, predicted):
    """One summary row per forecaster, all calculated by hand."""
    s = class_scores(actual, predicted)
    return {'forecaster': name,
            'accuracy': (actual == predicted).mean(),
            'macro_f1': s['f1'].mean(),
            'recall_very_poor': s.loc['Very Poor', 'recall'],
            'recall_severe': s.loc['Severe', 'recall'],
            'precision_severe': s.loc['Severe', 'precision']}

FORECASTERS = {                                   # name -> column in `val`
    'A: always ' + most_common: 'always_common',
    'B: typical for city + month': 'city_month',
    'D: category of the 7-day average': 'week_average',
    'C: persistence (tomorrow = today)': 'persistence',
}

results = pd.DataFrame([scorecard(name, val['actual'], val[col]) for name, col in FORECASTERS.items()]
                       ).set_index('forecaster')
results.round(3)
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>macro_f1</th>
      <th>recall_very_poor</th>
      <th>recall_severe</th>
      <th>precision_severe</th>
    </tr>
    <tr>
      <th>forecaster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A: always Satisfactory</th>
      <td>0.350</td>
      <td>0.086</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>B: typical for city + month</th>
      <td>0.531</td>
      <td>0.394</td>
      <td>0.783</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>D: category of the 7-day average</th>
      <td>0.660</td>
      <td>0.544</td>
      <td>0.587</td>
      <td>0.157</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th>C: persistence (tomorrow = today)</th>
      <td>0.710</td>
      <td>0.651</td>
      <td>0.668</td>
      <td>0.529</td>
      <td>0.519</td>
    </tr>
  </tbody>
</table>
</div>

**Findings:**
- **A (always Satisfactory)** is right on 35% of days, but its macro F1 is only 0.086: it never catches a bad day. Accuracy alone is misleading on imbalanced data.
- **B (typical for city + month)** reaches 53% and is the best of the four at Very Poor (recall 0.78): in Delhi in winter, Very Poor is usually the right call. It never predicts Severe, because Severe is never the *typical* category; an early sign that season information will help a model.
- **D (category of the 7-day average)** reaches macro F1 0.544, well above B but clearly below C. Weak on Severe: it catches only 8 of the 51 Severe days.
- **C (persistence)** is best: 71% accuracy, **macro F1 0.651**, 27 of the 51 Severe days caught. AQI is strongly autocorrelated (today's AQI correlates 0.86 with yesterday's), which persistence exploits directly.

**Macro F1 by information used:**

```
A  always Satisfactory           0.086   knows nothing
      │ +0.31   where and when matter
B  typical for city + month      0.394
      │ +0.15   the recent week beats the calendar
D  category of the 7-day average 0.544
      │ +0.11   today alone beats the whole week
C  persistence                   0.651   ◄── the bar
      │   ?     weather + trends + combining it all: the model's job
```

**Decision: persistence is the baseline.** Models are compared against the **strongest** simple rule; beating A, B or D would prove little.

### Why the 7-day average loses to today alone

The 51 days that were actually Severe, and what D forecast for them:

```python
cm_week = confusion_table(val['actual'], val['week_average'])
print('Severe days caught by D:', cm_week.loc['Severe', 'Severe'], 'of', cm_week.loc['Severe'].sum())
print('Days D called Severe:   ', cm_week['Severe'].sum())
cm_week.loc['Severe'].rename('what D forecast on the 51 Severe days')
```

**Output:**

```
Severe days caught by D: 8 of 51
Days D called Severe:    16
```

**Output:**

```
week_average
Good             0
Satisfactory     0
Moderate         1
Poor             5
Very Poor       37
Severe           8
Name: what D forecast on the 51 Severe days, dtype: int64
```

**Finding:** on the real Severe days, D mostly forecast **Very Poor** (37 of 51): averaging today with six older, usually calmer days **smooths away spikes**, and Severe days are spikes. D also reacts later than persistence (Delhi in section 3: 6 November instead of 3 November).

**Implication for the model:** the newest day carries the most information, so `aqi_today` stays the anchor feature; the weekly view is used **alongside** it, not instead of it, via the trend features (`aqi_roll7_std`, `aqi_dev7`).

## 9. Where persistence goes wrong

```python
changed = val['actual'] != val['persistence']        # the category is different tomorrow
print(f'Days that change category: {changed.sum()} of {len(val)} ({changed.mean():.0%})')
print('Persistence is right on every unchanged day and wrong on every changed day.')
```

**Output:**

```
Days that change category: 1478 of 5104 (29%)
Persistence is right on every unchanged day and wrong on every changed day.
```

Distance between actual and forecast category on wrong days (Good = 0 … Severe = 5):

```python
# How far off is persistence when it is wrong? Count the steps between the two categories.
step = {c: i for i, c in enumerate(AQI_CATEGORIES)}       # Good = 0 ... Severe = 5
distance = (val['actual'].map(step) - val['persistence'].map(step)).abs()
distance[changed].value_counts().sort_index().rename('wrong days')
```

**Output:**

```
1    1424
2      52
3       2
Name: wrong days, dtype: int64
```

Direction of change on those days. Rows: today's category; columns: steps worse (+) or better (−) tomorrow:

```python
# Which way does the category move on change days? (+1 = one category worse tomorrow)
direction = val['actual'].map(step) - val['persistence'].map(step)
pd.crosstab(val.loc[changed, 'persistence'], direction[changed],
            rownames=['today'], colnames=['steps tomorrow (+ = worse)']).reindex(AQI_CATEGORIES)
```

**Output:**

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>steps tomorrow (+ = worse)</th>
      <th>-3</th>
      <th>-2</th>
      <th>-1</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>today</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>105</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Satisfactory</th>
      <td>0</td>
      <td>0</td>
      <td>108</td>
      <td>273</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Moderate</th>
      <td>0</td>
      <td>2</td>
      <td>268</td>
      <td>230</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poor</th>
      <td>0</td>
      <td>9</td>
      <td>227</td>
      <td>87</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Very Poor</th>
      <td>0</td>
      <td>13</td>
      <td>85</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Severe</th>
      <td>1</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

**Findings:**
- **29% of validation days change category** (1,478 of 5,104). Persistence is wrong on all of them and right on the other 71%.
- **Errors are small:** 1,424 of the 1,478 wrong days (96%) are **one step** off. Persistence makes no large errors; it only misses transitions.
- **Both directions occur.** From Very Poor, tomorrow was one step better on 85 days and one step worse (Severe) on 21 days. A model has to separate these using today's weather and the recent trend.

A model therefore has to:
1. **Keep** the 71% of unchanged days right.
2. **Win** a share of the 29% change days, especially the jumps into Severe that persistence cannot anticipate.

## 10. Check against scikit-learn

The same metrics from `evaluate` in `utility.py` (scikit-learn), for all four forecasters. The cell raises an assertion error if any metric differs.

```python
# Cross-check: the hand-made numbers must match scikit-learn (utility.evaluate), for every forecaster
for name, col in FORECASTERS.items():
    check = evaluate(name, val['actual'], val[col])
    for key in ['accuracy', 'macro_f1', 'recall_very_poor', 'recall_severe', 'precision_severe']:
        assert np.isclose(results.loc[name, key], check[key]), (name, key)
print('Hand calculation matches scikit-learn for every metric of all', len(FORECASTERS), 'forecasters.')
```

**Output:**

```
Hand calculation matches scikit-learn for every metric of all 4 forecasters.
```

## 11. The bar for logistic regression

| Validation (2023) | A: always Satisfactory | B: city + month | D: 7-day average | **C: persistence** |
|---|---|---|---|---|
| Accuracy | 0.350 | 0.531 | 0.660 | **0.710** |
| **Macro F1** | 0.086 | 0.394 | 0.544 | **0.651** |
| Very Poor recall | 0.00 | 0.78 | 0.59 | **0.67** |
| Severe recall | 0.00 | 0.00 | 0.16 (8 of 51) | **0.53 (27 of 51)** |
| Severe precision | 0.00 | 0.00 | 0.50 | **0.52** |

The bar is **persistence**, which by definition gets none of the 29% change days right.

A model is worth keeping only if:
1. **Macro F1 is clearly above 0.651.** A gain of a few thousandths is noise.
2. **Severe recall rises without precision collapsing.** With only 51 Severe days, each day moves recall by about 2 points, so differences of one or two days are not meaningful. Predicting "Severe" on hundreds of days is not an improvement, even if more are caught.
3. **It gets a real share of the change days right**, the 29% where persistence is always wrong.
