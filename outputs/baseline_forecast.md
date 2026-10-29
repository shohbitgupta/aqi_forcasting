<!-- Exported from experiments/baseline_forecast.ipynb: every cell run top to bottom; generated, do not edit by hand -->

# Persistence Baseline: How Good Is "Tomorrow = Today"?

Before training any model, we need a **ruler**. A score like "71% accurate" means nothing on its own: it could be brilliant or useless, depending on how hard the problem is. So the first question is:

> **What score can we get without any machine learning at all?**

This notebook answers it with four "lazy forecasters" who know no ML, and keeps the strongest one as the **baseline** every model must beat.

Every number is calculated **by hand** with plain pandas: counting days, adding and dividing. Section 10 checks the results against scikit-learn.

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

A comparison is only fair if both sides play by the same rules, like a race on the same track:

| Rule | How |
|---|---|
| **Same days** | Drop the same 195 rows (empty trend features) that the models drop, in every split |
| **No peeking at the future** | A forecaster may only use what is known **today** |
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

## 3. Four lazy forecasters

Each forecaster uses exactly **one** source of information, in the simplest possible way:

| Forecaster | Rule | Information it uses |
|---|---|---|
| **A** | Always say the most common category | Nothing about the day (the **prior**) |
| **B** | Say the most common category for this city in this month | **Where and when** (called *climatology*) |
| **C, persistence** | Say tomorrow will be the same as today | **Today** |
| **D** | Say the category of the average AQI over the last 7 days | **The recent week** |

A and B "learn" their answer from the training years (2015–2022) only. C and D need no learning: they only look at the city's own recent days. Below, forecaster B's answers for Delhi, one per month.

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

Forecaster D takes `aqi_roll7_mean` (the average AQI of today and the 6 days before) and turns that number into its CPCB category with `aqi_to_category` (0–50 Good, 51–100 Satisfactory, …, 401–500 Severe). It asks: *does a calmer view of the whole week predict better than today alone?*

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

Now put the real answer and the four forecasts side by side, one row per validation day:

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

**Look at Delhi in early November 2023.** On 2 November the air jumped from Very Poor to Severe. Persistence said Very Poor: it only "sees" the change a day later. On 6 November the air improved to Very Poor, but persistence still said Severe. **Persistence is always one day late at every turn.**

The week average (D) is even later: the air was Severe from 2 November, but the 7-day average only reached Severe on **6 November**, because it still carries the calmer days from the week before. Keep this picture in mind; reacting late is the weakness a model has to fix.

## 4. Accuracy, by hand

Accuracy is the simplest score: **correct days ÷ all days**.

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

## 5. The confusion table, by hand

Accuracy hides *which* mistakes happen. The confusion table counts the days for every pair of (what really happened, what was forecast):

- **Rows** = what really happened tomorrow.
- **Columns** = what persistence forecast.
- **The diagonal** = correct days. Everything else is a mistake.

`pd.crosstab` simply counts the days in each cell.

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

Think of a **smoke alarm** for Severe days:

| Score | Question | From the table |
|---|---|---|
| **Recall** | Of all the real Severe days, how many did we catch? | diagonal ÷ **row** total |
| **Precision** | Of all the times we said "Severe", how often were we right? | diagonal ÷ **column** total |
| **F1** | One number that is high only when **both** are high | 2 × precision × recall ÷ (precision + recall) |

Missing a fire is dangerous (low recall). An alarm that rings all the time gets ignored (low precision).

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

The same three counts for each category. If a forecaster never predicts a category, its precision would divide by zero, so it is set to 0.

**Macro F1** is the plain average of the six F1 scores. Every category gets an equal vote, so rare Severe days count as much as common Satisfactory days.

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

## 8. Which lazy forecaster is the strongest?

The same hand-made scorecard for all four, ordered as a ladder: each step uses more recent information than the one before.

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
- **A (always Satisfactory)** is right on 35% of days, yet its macro F1 is only 0.086: it never catches a single bad day. This is why accuracy alone can fool us on imbalanced data.
- **B (typical for city + month)** reaches 53% and is the best of all four at Very Poor (recall 0.78): in Delhi in winter, "Very Poor" is usually a good guess. But it never says Severe, because Severe is never the *typical* answer. This is a first hint that season information will help a model.
- **D (category of the 7-day average)** reaches macro F1 0.544, well above the calendar (B) but clearly below today alone (C). Its weak spot is Severe: it catches only 8 of the 51 Severe days.
- **C (persistence)** wins: 71% accuracy, **macro F1 0.651**, and it catches 27 of the 51 Severe days. Air pollution has memory (today's AQI correlates 0.86 with yesterday's), and persistence gets that memory for free.

**The ladder: what each kind of information is worth (macro F1)**

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

**Decision: persistence is the baseline.** We always compare against the **strongest** simple rule. Beating A, B or D would prove very little.

### Why does the week average lose to today alone?

Look at the 51 days that really turned out Severe, and what D forecast for them:

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

**Finding:** on the real Severe days, D mostly said **Very Poor** (37 of 51). An average mixes today with six older, usually calmer days, so it **smooths away the spikes**, and Severe days are spikes. It also reacts later than persistence (Delhi in section 3: 6 November instead of 3 November).

**What this means for the model:** the newest day carries the most information, so `aqi_today` stays the anchor feature. The weekly view is still useful **next to** it, not instead of it: that is exactly how the trend features (`aqi_roll7_std`, `aqi_dev7`) are used in the model.

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

How far off is it when it is wrong? Count the steps between the real and the forecast category (Good = 0 … Severe = 5):

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

And which way does the category move on those days? Rows are today's category; columns are how many steps worse (+) or better (−) tomorrow turned out:

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
- **29% of validation days change category** (1,478 of 5,104). Persistence gets all of them wrong, and all of the other 71% right.
- **The mistakes are small:** 1,424 of the 1,478 wrong days (96%) are just **one step** off. Persistence never makes wild errors; it only misses the moves.
- **Both directions happen.** From Very Poor, tomorrow was one step better on 85 days and one step worse (Severe) on 21 days. A model has to tell these apart from today's weather and the recent trend.

So a good model must do two things at once:
1. **Keep** the 71% of easy days right: it must not break what persistence already gets right.
2. **Win** a share of the 29% change days, especially the jumps into Severe that persistence never sees coming.

## 10. Check against scikit-learn

The same numbers from `evaluate` in `utility.py`, which uses scikit-learn, for all four forecasters. If any metric differed, this cell would stop with an error.

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

The bar is **persistence**. It gets none of the 29% change days right, by definition.

A model is worth keeping only if:
1. **Macro F1 is clearly above 0.651.** A gain of a few thousandths is noise.
2. **Severe recall rises without precision collapsing.** With only 51 Severe days, each day moves recall by about 2 points, so ignore differences of one or two days. A model that says "Severe" on hundreds of days is not better, even if it catches more.
3. **It gets a real share of the change days right**, the 29% where persistence is always wrong.
