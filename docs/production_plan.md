# Production plan: next-day AQI forecast on demand

**Status:** plan. Phase 1 is the course deliverable; the other phases come after deployment.

**Goal:** a user picks one of the 14 cities in the application and gets tomorrow's air quality category for that city, with the chance of a Severe and of a Dangerous day.

---

## The idea in one page

Three pieces, each with one job:

| Piece | Job | Phase |
|---|---|---|
| **Prediction service** | Takes one city-day's 27 input fields as JSON, returns the forecast. Knows nothing about where the data comes from | 1 (course deliverable) |
| **Data store and daily refresh** | Fetches each new CPCB bulletin and the weather once a day, keeps the history, prepares each city's 27 fields | 2 |
| **Application** | The user picks a city; the application reads that city's prepared fields, calls the prediction service and shows the answer | 3 |

```
Go-live (once):   fetch past bulletins for all 14 cities → data store
Every day:        about 4:30 PM IST, after the bulletin → fetch today (14 cities)
                  → data store → build each city's 27 fields → ready
User request:     city → read its 27 fields → prediction service → forecast + the date it is for
```

**Why the data is fetched once a day and not for each user.** Everyone asking about Delhi on the same evening gets the same answer, because the inputs only change when a new bulletin comes out.

| | Fetch live on every request | Fetch once a day, then look up (chosen) |
|---|---|---|
| Response time | Several seconds (8 PDF downloads and parsing) | Milliseconds |
| Load on CPCB and Open-Meteo | Grows with every user | One fetch a day, however many users |
| If CPCB is slow or down | Every user gets an error | Users still get the latest prepared data |

### Which day "tomorrow" is

CPCB publishes its bulletin at about 4 PM: the average of the past 24 hours. A forecast is always for the **next bulletin** after the latest one we have, so it depends on when the user asks:

| User asks at | Latest bulletin | Forecast is for |
|---|---|---|
| 10 AM, 29 Oct | 28 Oct, 4 PM | **29 Oct** (today's bulletin, not out yet) |
| 6 PM, 29 Oct | 29 Oct, 4 PM | **30 Oct** |

The application always shows the date, for example: *"Forecast for 30 Oct (24 hours to 4 PM), based on the 29 Oct bulletin."*

---

## Phase 1: the prediction service (course deliverable)

- `scripts/predict.py`: loads `models/xgb_final.bin` once, takes a city-day's 27 fields as JSON, returns:
  - the most likely category;
  - P(Severe) and P(Dangerous) (Very Poor or Severe);
  - an alert flag, raised when P(Dangerous) or P(Severe) passes a threshold **set in the service's configuration**, not in the model ([`model_decisions.md`](model_decisions.md), Decision 11).
- Web framework: FastAPI or Flask. Packaged with Docker.
- **Stateless:** it stores nothing and looks up nothing. This keeps it simple, testable with one request, and exactly matched to how the model was trained.
- Rejects any city the model was not trained on (the list is saved in the model file).

For demonstrations, a script builds the 27 fields for a real day by downloading that day's CPCB bulletin, the 7 days before it, and that day's weather. It was tested on 28 October 2026 for all 14 cities.

---

## Phase 2: data store and daily refresh

### Go-live: fill the history once

- **Minimum:** the last **8 days** of bulletins for all 14 cities. The features use today and the 7 days before (the "deviation from last week" compares today with the average of the previous 7 days), so the first forecast needs 8 bulletins.
- **Recommended:** every bulletin from **1 January 2026**. The training data ends in December 2025, so this fills the gap and becomes the data for monitoring and the next retraining. CPCB keeps old bulletins online, so this is one extra run.
- Weather is needed only for **today** (past weather is not a model input), so the backfill needs bulletins only.

### The daily refresh

Runs every day at about **4:30 PM IST**. If the bulletin is not out yet, it retries every 30 minutes until about 7 PM.

| Step | What happens |
|---|---|
| 1. Catch up | Check the last 7 days in the store. Fetch any missing bulletin, so a day missed by a failed run repairs itself |
| 2. Fetch | Today's CPCB bulletin (14 cities: AQI, category, prominent pollutants, stations) and today's weather from Open-Meteo |
| 3. Store | Add one row per city per day. Re-running a day replaces its rows, so a retry never creates duplicates |
| 4. Build features | From the stored history, compute each city's 27 fields with the same code as in training: 7-day mean and spread, change since yesterday, deviation from last week, pollutant flags, month |
| 5. Save | Store the 27 fields per city with the bulletin date. The application reads these |
| 6. Log a forecast | Run the model once per city and log the result, for monitoring (Phase 4), whether or not a user asks |

### What is stored

| Table | One row per | Contents |
|---|---|---|
| Daily data | city and day | AQI, category, prominent pollutants, stations reporting, the 14 weather values, source, time fetched |
| Prepared features | city and bulletin date | The 27 model fields, and the date the forecast is for |
| Forecast log | city and bulletin date | Forecast, probabilities, and later the real category from the next bulletin |

All three are small (14 rows a day each). Keep everything.

### When something is missing

The same rules as in training, so the model never sees anything it was not trained on:

| Problem | What happens |
|---|---|
| A city is missing from today's bulletin | Use the last known AQI for up to 3 days, with `aqi_imputed = 1` |
| Missing for more than 3 days | No forecast for that city; the application says so; raise an alert |
| Bulletin not published by 7 PM | Keep serving the previous day's data (its forecast date is shown); raise an alert. Never present old data as new |
| Weather download fails | Retry; if it still fails, no new features that day (weather is never guessed) |
| A value outside the training range (for example AQI above 500) | Keep the forecast, flag it for review |

---

## Phase 3: the application

1. The user picks a city from a **list of the 14 cities** (a list, not free text).
2. The application reads that city's latest prepared fields from the store.
3. It sends them to the prediction service and gets the forecast back.
4. It shows the category, the chance of a Severe and of a Dangerous day, any alert, and **the date the forecast is for**.

If there is no forecast for the city that day, the application says so instead of showing an old or guessed value.

---

## Two differences between training and production to check

| Difference | Why it matters | How to check |
|---|---|---|
| **Today's weather is partly a forecast at 4:30 PM.** In training, "today's weather" was the full real day (midnight to midnight); at 4:30 PM the evening hours are still forecast values | The model may see slightly different weather than it learned from | For 4 weeks, store the 4:30 PM weather and compare it with the same days' final values a week later |
| **Different weather source.** Training used Open-Meteo's historical archive; production uses its forecast API | The two can differ slightly for the same day | Same comparison. If the gap is large, retrain on the forecast API's history |

If both gaps are small, nothing changes. If not, the fix is to retrain on data collected the same way as in production.

---

## Phase 4: monitoring and retraining

### Monitoring: every forecast is checked the next day

The daily refresh logs a forecast for every city (step 6), and the next bulletin is its answer, so the model's real performance is known every day at no cost.

| What | How often | Warning sign |
|---|---|---|
| Overall score (macro F1), model vs "tomorrow = today" | Rolling 30 days | Model behind the simple rule for 4 weeks in a row |
| Dangerous days warned about, and warnings that were right | Rolling 30 days, mostly in winter | Clearly below the validation level (about 81% and 77%) |
| Severe days caught, missed and false alarms | Each winter season | Large change from validation |
| Data checks (missing cities, values out of range, imputed days, late bulletins) | Daily | More than one city missing, or repeated imputation |

### Retraining

- **When:** once a year, before the winter season (by September), or earlier if monitoring shows a warning sign.
- **How:** the same process as now. Rolling validation on the past five years, then a final test on the most recent year not used for any decision.
- **Rule:** a new model replaces the current one only if it beats both the simple rule and the current model on the same test year.

### The alert threshold

- Each user group (hospital, school, public app) can have its own threshold.
- A threshold is always chosen on past validation years, **never on the test year**.

---

## Deployment on AWS within the free tier

| Piece | AWS service | Free allowance (per month) | Our use |
|---|---|---|---|
| Prediction service | **Lambda**, from a container image, with a Function URL (public HTTPS link) | 1 million requests, 400,000 GB-seconds (always free) | User requests plus 14 logged forecasts a day |
| Daily refresh | **Lambda**, started by **EventBridge Scheduler** at about 4:30 PM IST | Scheduler: 14 million invocations | About 30 a month, plus retries |
| Data store | **DynamoDB** | 25 GB (always free) | 3 small tables, 14 rows a day each |
| Container images | **ECR** | 500 MB for the first year, then about $0.10 per GB | One or two images |

**Practical points:**
- **Use a container image, not a zip upload.** A zip upload is limited to 250 MB unpacked, and our libraries (scipy, pandas, scikit-learn, numpy, xgboost) are close to that. Install `xgboost-cpu` in the image: the same library without the GPU parts, so the image stays smaller.
- **Set a billing alert** (for example $1) before deploying anything.
- **Account type matters.** Accounts created before 15 July 2025 keep the old 12-month free tier. Newer accounts on the Free plan get up to $200 in credits but close after 6 months unless switched to the Paid plan. Lambda and DynamoDB stay free on both.
- **Cold starts:** the first request after a quiet period takes a few seconds while Lambda loads the libraries and the model.
- **Open-Meteo's free API is for non-commercial use.** A commercial launch needs their paid plan.

---

## Open questions

1. **Application:** web page, mobile app, or chat bot? Who are the first users?
2. **Alert thresholds:** one shared setting, or one per user group? Who signs them off?
3. **AWS account:** created before or after 15 July 2025 (decides the free tier rules)?
4. **History:** backfill from 1 January 2026 (recommended) or only the minimum 8 days?
