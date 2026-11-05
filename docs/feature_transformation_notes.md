# Feature transformation notes

Why logistic regression gets four transformed features in this project, and why the final model does not.

**Where these are used:**

| Model | Transforms | Where |
|---|---|---|
| Logistic regression | `pressure_rel`, `precip_log`, `month_sin`, `month_cos` instead of `pressure_mean`, `precip_sum`, `month` | `add_lr_features` in [`scripts/utility.py`](../scripts/utility.py), used in `experiments/lr_train_baseline.ipynb` |
| Decision tree, random forest, XGBoost | **None:** the raw columns | `TREE_FEATURES` in `notebooks/aqi_prediction.ipynb` |
| **The final model and the web service** (XGBoost) | **None:** the raw columns | `NUMERIC` in [`scripts/train.py`](../scripts/train.py); the API takes `month`, `pressure_mean` and `precip_sum` as they are |

**Why only logistic regression:** it gives each feature one weight and adds them up, so it can only use a feature whose effect is roughly "more of X → more (or less) likely". Tree models split on thresholds ("month ≥ 11", then "city is Delhi", then "pressure > 985"), so they find these patterns in the raw columns. The data has many such non-linear patterns, which is one reason a tree model was chosen in the end ([`model_decisions.md`](model_decisions.md), Decision 12).

**How the numbers below were measured:** on the training years only (2015–2022, 33,731 city-days), as `rho` = Spearman correlation with tomorrow's category numbered 0 (Good) to 5 (Severe). The same check is in `notebooks/aqi_prediction.ipynb`, section 4.

| Symptom | Fix | In this project |
|---|---|---|
| Cyclical feature with a false jump at the wrap-around | [Cyclical (sin/cos) encoding](#1-cyclical-encoding-month) | `month` → `month_sin`, `month_cos` |
| Very skewed feature with a long tail | [Log transform](#2-log-transform-rain) | `precip_sum` → `precip_log` |
| Feature that mostly says which city it is | [Subtract the city's normal](#3-subtract-the-citys-normal-pressure) | `pressure_mean` → `pressure_rel` |
| Direction plus strength | [Vector parts](#4-vector-parts-wind) | wind → `wind_u_mean`, `wind_v_mean` (all models) |
| Categories with no order | [One-hot encoding](#5-one-hot-encoding-city) | `city` → 14 columns (all models) |

---

## 1. Cyclical encoding: month

### The problem

Month as a plain number (1–12) has two problems for logistic regression:

1. **A false jump at the wrap-around.** December and January are neighbours, but as numbers they are 12 and 1, the furthest apart any two months can be.
2. **A shape one weight cannot follow.** Air quality is bad at **both ends** of the year and clean in the middle, so "a higher month number → worse air" is wrong for half the year.

| | Jan | … | Jul | … | Dec |
|---|---|---|---|---|---|
| Month number | 1 | | 7 | | 12 |
| Air quality | **Bad** | | Clean | | **Bad** |

It is **not** a problem of scale: standardising the month would leave both problems in place.

### The fix

Place each month on a circle with two coordinates:

```python
angle = 2 * np.pi * df['month'] / 12
df['month_sin'] = np.sin(angle)
df['month_cos'] = np.cos(angle)
```

| Month | `month_cos` | `month_sin` |
|---|---|---|
| Jan | 0.87 | 0.50 |
| Feb | 0.50 | 0.87 |
| Apr | −0.50 | 0.87 |
| Jun | −1.00 | 0.00 |
| Jul | −0.87 | −0.50 |
| Oct | 0.50 | −0.87 |
| Nov | 0.87 | −0.50 |
| Dec | 1.00 | 0.00 |

- December (1.00, 0.00) and January (0.87, 0.50) are now neighbours.
- `month_cos` is high in winter and low in the monsoon, which matches the pollution pattern in a straight line.
- `month_sin` separates months that share the same `month_cos`: February (+0.87, air improving) and October (−0.87, crop burning starts, air getting worse) both have `month_cos` = 0.50. This is why both columns are kept: with `month_cos` alone, February and October look the same.

### The result

| Feature | rho |
|---|---|
| `month` (raw number) | **0.00** |
| `month_cos` | **0.48** |
| `month_sin` | 0.21 (weak alone, useful together with `month_cos`) |

---

## 2. Log transform: rain

**Problem:** rain (`precip_sum`) is extremely skewed: **57% of days have 0 mm**, but a few reach 251 mm (skew 7.8). Logistic regression's single weight gets pulled by those few storms, and the difference between a dry day and a day with light rain is lost.

**Fix:** `precip_log = np.log1p(precip_sum)`, i.e. log(1 + rain), which keeps 0 mm at 0.

**Result:** the skew drops from **7.8 to 1.6**. Rain is one of the strongest weather signals (rho **−0.60**: more rain today, cleaner air tomorrow), and the log keeps that unchanged (−0.60 after), because a log does not change the order of the values. Only the extreme days stop dominating.

---

## 3. Subtract the city's normal: pressure

**Problem:** air pressure is mostly **altitude**. The training-year normals run from 912 hPa in Bengaluru (about 900 m high) to 1008 hPa in Mumbai (sea level), a gap of 97 hPa. Within one city, pressure changes by under 1 hPa on a typical day and varies by about 6 hPa (standard deviation) over the years, so raw pressure mostly tells the model which city a row is from, which the `city` columns already say.

**Fix:** subtract each city's mean pressure, computed on the **training years only** (otherwise information from 2023–2025 leaks in):

```python
city_pressure = df_train.groupby('city')['pressure_mean'].mean()
df['pressure_rel'] = df['pressure_mean'] - df['city'].map(city_pressure)
```

**Result:**

| Feature | rho |
|---|---|
| `pressure_mean` (raw) | 0.18 |
| `pressure_rel` | **0.52** |

Higher pressure than the city's normal goes with calm, stagnant air, in which pollution builds up.

**Read with care:** much of this signal is the **season**. `pressure_rel` correlates at 0.81 with `month_cos` and −0.83 with `temp_min`: pressure is above normal in winter, the polluted season. The three stay together in logistic regression because each still adds a little on its own (VIF below 10, `notebooks/aqi_prediction.ipynb` section 5), but their separate weights should not be read as three independent effects.

---

## 4. Vector parts: wind

**Problem:** wind is a direction plus a strength. The angle has the same false jump as the month (359° → 0°), and a direction means little without its strength: a calm day "from the north" is not a strong north wind.

**Fix:** split the wind into an east-west part (u) and a north-south part (v). This is done once, when the weather is downloaded, in [`code/download_weather.py`](../code/download_weather.py):

```python
rad = np.deg2rad(h['wind_direction_10m'])
h['wind_u'] = -h['wind_speed_10m'] * np.sin(rad)     # east-west part
h['wind_v'] = -h['wind_speed_10m'] * np.cos(rad)     # north-south part
```

The minus signs are there because the reported direction is where the wind comes **from**, while u and v point where it blows **to**. The hourly u and v are then averaged into the day's `wind_u_mean` and `wind_v_mean`. This is also the correct way to average directions: a plain mean of 350° and 10° gives 180°, the opposite direction.

Unlike sections 1–3, **every model uses u and v**, including the final model. The EDA dropped the raw angle `wind_dir_mean`.

---

## 5. One-hot encoding: city

**Problem:** the 14 cities have no order. Numbered (Bengaluru = 0, Chennai = 1, Delhi = 2, …), they would tell logistic regression that one city is "more" than another.

**Fix:** one 0/1 column per city: Delhi gets `city=Delhi` = 1 and 0 in the other 13.

**In this project** a `DictVectorizer` does this inside each model's pipeline, for logistic regression and for the final XGBoost model ([`scripts/train.py`](../scripts/train.py)). That is why the web service accepts the city as text, `"city": "Delhi"`, and the model turns it into the 14 columns itself.

---

## The other engineered features

The trend features (`aqi_roll7_mean`, `aqi_roll7_std`, `aqi_diff1`, `aqi_dev7`), the pollutant flags (`has_pm25`, …) and `aqi_imputed` are used by every model. They are explained in `notebooks/eda.ipynb`, `notebooks/aqi_prediction.ipynb` (sections 2 and 5) and [`EDA_Decision_Document.pdf`](EDA_Decision_Document.pdf).
