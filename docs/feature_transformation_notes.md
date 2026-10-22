# Feature transformation notes

Reusable notes on making a feature usable by the model. Each technique is explained once, with the example from the AQI forecasting project (`notebooks/aqi_prediction.ipynb`, section 4) as proof.

**The general idea:** a feature can carry real signal but in a form the model cannot use. Linear models (logistic and linear regression), distance-based models (KNN, SVM) and neural networks are the ones affected. Tree models (decision trees, random forest, XGBoost) split on thresholds, so they generally use the raw feature as it is.

| Symptom | Fix | Example in this project |
|---|---|---|
| Cyclical feature with a false jump at the wrap-around | [Cyclical (sin/cos) encoding](#1-cyclical-encoding) | `month` → `month_sin`, `month_cos` |
| Very skewed feature with a long tail | [Log transform](#2-log-transform) | `precip_sum` → `precip_log` |
| Feature mixes a real effect with a group's baseline | [Subtract the group's normal](#3-subtract-the-groups-normal) | `pressure_mean` → `pressure_rel` |
| Direction plus strength (an angle) | [Split into vector parts](#4-vector-parts-for-directions) | wind → `wind_u_mean`, `wind_v_mean` |
| Categories with no order | [One-hot encoding](#5-one-hot-encoding) | `city` |

---

## 1. Cyclical encoding

Also called **cyclical feature encoding** or **sine/cosine (trigonometric) encoding**. In time-series work, the general form is **Fourier features** (Fourier terms).

### The problem

A feature that **repeats in a cycle**, where the end of the cycle sits right next to the start. As a plain number, the model sees two problems:

1. **A false jump at the wrap-around.** 31 December and 1 January are one day apart, but as month numbers they are 12 and 1, the furthest apart any two months can be.
2. **A shape a linear model cannot follow.** A linear model gives the feature one weight, so it can only learn "higher number → steadily more (or less) likely". A cyclical effect usually rises and falls within the cycle.

It is **not** a problem of size: standardising the month (subtract the mean, divide by the standard deviation) would leave both problems in place.

### The fix

Place each value on a circle with two coordinates:

```python
angle = 2 * np.pi * value / period
df['x_sin'] = np.sin(angle)
df['x_cos'] = np.cos(angle)
```

`period` is the length of one full cycle.

### Example: month in the AQI project

Air quality is bad at **both ends** of the year and clean in the middle:

| | Jan | … | Jul | … | Dec |
|---|---|---|---|---|---|
| Month number | 1 | | 7 | | 12 |
| Air quality | **Bad** | | Clean | | **Bad** |

On the circle:

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
- `month_sin` separates months that share the same `month_cos`: February (+0.87, air improving) and October (−0.87, crop burning starts, air getting worse) both have `month_cos` = 0.50.

**Proof (training years, Spearman correlation with tomorrow's category):**

| Feature | rho |
|---|---|
| `month` (raw number) | **0.00** |
| `month_cos` | **0.48** |
| `month_sin` | 0.21 (weak alone, useful together with `month_cos`) |

### Where it applies

| Feature | Period | The false jump |
|---|---|---|
| Month | 12 | December (12) → January (1) |
| Hour of day | 24 | 23:00 → 00:00 |
| Day of week | 7 | Sunday → Monday |
| Day of year | 365 (or 365.25) | 31 Dec → 1 Jan |
| Minute of hour | 60 | :59 → :00 |
| Wind / compass direction, any angle | 360 | 359° → 0° |

### Rules

1. **Always use both sin and cos.** With only one, two different points get the same value (with cos only, February = October).
2. **Use the right period.** 12 for months numbered 1–12, 24 for hours numbered 0–23. With the wrong period, the circle does not close.
3. **Only for truly cyclical features.** Age, price and year do not wrap around.
4. **Needed for** linear and logistic regression, KNN, SVM and neural networks.
5. **Not needed for tree models.** A tree can split on "month ≥ 11 or month ≤ 1" directly. In this project, tree models keep the raw `month`.

### Alternatives

| Option | Columns | Use when |
|---|---|---|
| **Sin/cos** (one pair) | 2 | The cycle has one smooth peak and one trough (winter vs monsoon) |
| **One-hot** (one column per value) | period − 1 | Few values (7 weekdays, 12 months) and each value may behave differently, e.g. a sharp October jump |
| **Fourier features** (several sin/cos pairs, k = 1, 2, 3, …) | 2k | The cycle has several peaks, e.g. traffic with a morning and an evening rush hour |
| **Periodic splines** (`SplineTransformer(extrapolation='periodic')` in scikit-learn) | a few | A smooth but more flexible shape |

Fourier features with harmonic *k*: `sin(2πk · value / period)` and `cos(2πk · value / period)`. With k = 1 this is plain cyclical encoding.

### Check that it worked

Compare the feature's correlation with the target before and after (as in the proof table above). If the encoded version is no stronger, the feature may not be cyclical in its effect, or one-hot may fit better.

---

## 2. Log transform

**Problem:** a very skewed feature: most values small, a few very large. A linear model's single weight gets pulled by the few extreme values, and the differences among the common small values are lost.

**Fix:** `np.log1p(x)`, i.e. log(1 + x), which also handles zeros. Only for values ≥ 0.

**Example:** rain (`precip_sum`) is 0 mm on most days but reaches 251 mm. `precip_log` cuts the skew from **7.8 to 1.6**. The rank correlation with the target does not change (a log keeps the order), but the linear model is no longer dominated by a handful of storms.

**Alternative:** a 0/1 flag (`rained`) when *whether* it happened matters more than *how much*.

**Rule of thumb:** consider it when |skew| > 1 and the feature goes into a linear model.

---

## 3. Subtract the group's normal

**Problem:** a feature mixes a real effect with a fixed baseline that differs by group. The model then mostly learns *which group* a row belongs to.

**Fix:** subtract each group's mean, computed on the **training data only**:

```python
group_mean = df_train.groupby('city')['pressure_mean'].mean()
df['pressure_rel'] = df['pressure_mean'] - df['city'].map(group_mean)
```

**Example:** air pressure is mostly altitude. Bengaluru's normal is far lower than Delhi's (city normals differ by up to ~97 hPa), so raw pressure mostly says "this is Bengaluru". After subtracting each city's normal, the weather effect appears: higher pressure than usual means calm, stagnant air, and pollution builds up.

| Feature | rho with tomorrow's category |
|---|---|
| `pressure_mean` (raw) | 0.18 |
| `pressure_rel` | **0.52** |

**Rule:** compute the group means on the training split only; otherwise information from the validation and test years leaks in.

---

## 4. Vector parts for directions

**Problem:** a direction plus a strength (wind, movement). The angle has the 359° → 0° jump (see section 1), and a direction means little without its strength: a calm day "from the north" is not the same as a storm from the north.

**Fix:** split it into vector components, which is cyclical encoding scaled by the strength:

```python
rad = np.radians(direction)
u = -speed * np.sin(rad)     # east-west part
v = -speed * np.cos(rad)     # north-south part
```

The minus signs follow the meteorological convention: the reported direction is where the wind comes **from**, while u and v point where it blows **to**. For an angle that is already "towards" (a heading, for example), drop the minus signs.

**Example:** the dataset's `wind_u_mean` and `wind_v_mean` are built exactly this way in `code/download_weather.py`, so the EDA dropped the raw angle `wind_dir_mean`. Averaging u and v over a day is also the correct way to average directions: a plain mean of 350° and 10° gives 180°, the opposite direction.

---

## 5. One-hot encoding

**Problem:** categories with no natural order (city, product type). Numbering them (Bengaluru = 0, Delhi = 3) tells a linear model that cities have an order and a size.

**Fix:** one 0/1 column per category.

**Example:** `city` → 14 columns. Delhi gets `city_Delhi = 1` and 0 everywhere else.

**Watch:** with many categories (hundreds), one-hot creates many sparse columns. Consider grouping rare categories into "other", or target encoding (computed on training data only).

---

## Checklist for a new problem

1. Does the feature **repeat in a cycle**? → cyclical encoding.
2. Is it **very skewed** (|skew| > 1) and going into a linear model? → log transform or a 0/1 flag.
3. Does its level mostly depend on a **group** (city, store, sensor)? → subtract the group's normal (training data only).
4. Is it a **direction with a strength**? → vector components.
5. Is it a **category with no order**? → one-hot.
6. **Check:** compare the feature's correlation with the target before and after. Keep the transform only if it helps.
7. **Tree models:** usually skip all of the above and use the raw features.
