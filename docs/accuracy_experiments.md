# Accuracy experiments: can the final model be right more often?

**Question:** can simple changes raise **overall accuracy** (the share of days the forecast category is the one CPCB reports)?
**Answer:** only slightly. The model is at the limit of what this data can tell; see the [conclusion](#conclusion).

Notebooks: [`experiments/accuracy_experiments.ipynb`](../experiments/accuracy_experiments.ipynb) (exported to [`outputs/accuracy_experiments.md`](../outputs/accuracy_experiments.md)) and [`experiments/weather_window.ipynb`](../experiments/weather_window.ipynb).

## What was tried

| Idea | Why |
|---|---|
| **Weight recent years more** (a day counts half as much for every 3 years back) | the air in several cities is cleaner than in 2015-2018 |
| **Forecast the AQI number**, then turn it into the category | the model would know that 291 is almost 301 |
| **AQI 2 and 3 days ago** as extra inputs | a longer memory of the trend |
| **Perfect tomorrow's weather** (the weather that really happened; cannot be deployed) | the most any weather forecast could ever add |

Same rules as every model: chosen with rolling validation on 2019-2023, then scored once on 2025, always next to persistence ("tomorrow = today"). One extra check no model had seen: **31 live days in October 2026**, built from the CPCB bulletins exactly as the web service does ([`data/processed/live_check_2026_10.csv`](../data/processed/live_check_2026_10.csv)).

## Results (accuracy)

| | Validation 2019-2023 | Test 2025 | Live Oct 2026 |
|---|---|---|---|
| Persistence | 68.1% | 73.3% | 69.1% |
| **Current model** | **71.3%** | **75.1%** | **71.4%** |
| Recent years weighted | 71.5% | 75.8% | 72.3% |
| Recent years weighted, also trained on 2025 | - | - | 73.0% |
| AQI number, then category | 71.0% | 74.2% | 70.7% |
| AQI 2-3 days ago | 71.4% | - | - |
| Perfect tomorrow's weather (ceiling) | 72.4% | - | - |

## Findings

1. **Weighting recent years helps a little, every time**: +0.2 points in validation, +0.7 on 2025, +0.9 live; about +1.6 live when 2025 is added to training.
2. **Forecasting the AQI number does not help**: it catches more category changes but misses more steady days.
3. **A longer memory adds nothing**, and **even perfect tomorrow's weather adds only about 1 point**.
4. **The limit is the data, not the model.** With these inputs, tomorrow's AQI cannot be forecast to better than about 20-24 points; a forecast that far off gets the category right on about 70% of days, which is where every candidate lands. Reaching 85% would need tomorrow's AQI to within about 10 points.

## Why Poor, Very Poor and Severe get confused

The model mixes up the three high categories most (2025: 63% of Poor, 71% of Very Poor and 58% of Severe days caught; almost every miss is a neighbouring category). Four possible causes were tested (notebook section 9):

| Test | Result | Meaning |
|---|---|---|
| **How much tomorrow moves** on days with AQI above 200 | typically 34 points; 50+ on 34% of days, while a category is 100 points wide. From AQI 401-450 today, tomorrow is Severe 51%, Very Poor 43% | today's AQI cannot pin down tomorrow's band |
| **Near-identical days** (same city, most similar in all inputs) | end in a different category on 32% of high days; a Severe day's look-alike was Severe only 46% of the time, Very Poor 46% | **the inputs cannot separate these days, whatever the model** |
| **More training data** (25% to 100% of 2015-2022, scored on 2023) | Poor and Very Poor flat (61-63%, 72-76%); Severe 35% up to half the data, 47-49% from three quarters on | not a shortage of rows |
| **A much bigger model** (depth 8 instead of 3) | 100% on the training years, no better on 2023 (Very Poor 73% vs 75%, Severe 43% vs 47%) | not the model: it memorises noise |
| **Few monitoring stations** (2025 high days) | error 39% with 1-3 stations, 25% with 16+ (partly a city effect) | measurement noise adds to it |

**Conclusion: a data issue of missing information**, plus some measurement noise; not the model, and mostly not too few examples. A gain needs new information about what drives a jump on polluted days: tomorrow's weather forecast (wind, cold-night inversions), crop-fire activity or upwind AQI, festival dates.

## Weather on the bulletin's window

The bulletin of day D averages **4 PM on D-1 to 4 PM on D**, but the weather inputs cover the **calendar day** D, so their last 8 hours fall in the window being forecast. The same Open-Meteo hourly data was re-aggregated over each bulletin's window ([`code/download_weather_window.py`](../code/download_weather_window.py)) and compared, on the five NCR cities Delhi, Noida, Greater Noida, Ghaziabad and Gurugram ([`experiments/weather_window.ipynb`](../experiments/weather_window.ipynb)):

| Weather inputs | Validation 2019-2023 | Right on change days |
|---|---|---|
| **Calendar day (current)** | **66.0%** | 37.8% |
| Bulletin window | 64.5% (lower in 5 of 5 years) | 33.9% |
| Bulletin window + the target window's weather (ceiling, not deployable) | 67.2% | 41.0% |

- **The 8 evening hours carry real information** about the next bulletin: dropping them costs 1.5 points, mostly on change days. They work as a very short-range forecast.
- **Not a leak in practice:** live, at about 5 PM, those hours come from Open-Meteo's short-range estimates, so the service keeps most of the benefit. **The calendar-day weather stays.**
- The 2025 test and the live days move by about half a point either way, within noise.

## Decision

| | Keep the current model | Retrain now (2015-2025, recent years weighted) |
|---|---|---|
| Gain | - | about +0.5 to +1.5 points |
| Cost | none | 2025 stops being an untouched test year; scores, README and app numbers need updating; redeploy |

**Keep the current model.** The gain is about one more right day in a hundred, while 2025 is the cleanest evidence that the model beats persistence. **Next yearly retrain:** once 2026 is complete, train on 2015-2025 with recent years weighted and test on 2026, which gives the gain and a fresh test year.

## Conclusion

**The model is at the limit of this data; the limit is information, not the model or the amount of data.**

| Evidence | Result |
|---|---|
| Four model types, tuned | the best all reach about 71-72% in validation |
| A much bigger model | memorises the training days, no better on new ones |
| More training data | flat for Poor and Very Poor |
| Recency weights, AQI-number target, longer memory, bulletin-window weather | between -1.5 and +0.7 points |
| Perfect tomorrow's weather (not deployable) | only about +1 point |
| Near-identical days | end in a different category 32% of the time |

Tomorrow's AQI depends on things the inputs do not contain: tomorrow's weather, crop-fire smoke, festivals, which pollutant will dominate and which stations will report. On polluted days it moves about 34 points a day in a 100-point band, and the bulletin's averaging over stations adds measurement noise. So days that look the same end differently, and no model can separate them.

The current model is a sound result: 75.1% accuracy on the 2025 test (persistence 73.3%), 83% of Very Poor or Severe days caught, and a macro ROC AUC of 0.952.

**The way forward is new information, and the expected gains are modest:**

| New source | Expected effect |
|---|---|
| Archive of past weather forecasts (Open-Meteo), for the target window | up to about +1 point overall, more on change days |
| Crop-fire counts (NASA FIRMS) or upwind AQI | NCR smoke episodes in October-November |
| Festival dates (Diwali) | a few high-impact days a year |
