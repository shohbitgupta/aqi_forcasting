# Modelling decisions: plain-language summary

**Project:** forecast tomorrow's air quality category (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) for 14 Indian cities.

**Where we are:** the data is prepared, the simple forecasts are measured, and four machine-learning models (logistic regression, a decision tree, a random forest and XGBoost) have been tested, first on 2023 and then on five separate years. This page explains the decisions so far and why they were taken. The full side-by-side numbers are in [`model_comparison.md`](model_comparison.md).

Most single-year results come from **2023** (5,104 city-days), a year the models never saw while learning. 2023 had **51 Severe days**. From Decision 7 on, every model is also checked on **five years (2019–2023)**, because one year alone turned out to be misleading.

---

## Summary in one page

| # | Decision | Why it matters |
|---|---|---|
| 1 | The model must beat **"tomorrow will be the same as today"** | This simple rule is already right 71% of the time. A model is only useful if it does better |
| 2 | Our first model is a simple, explainable one (**logistic regression**) | It shows which factors drive the forecast before we try more complex models |
| 3 | We do **not** force the model to predict rare days more often | Forcing it caused 85 false Severe alarms in one year |
| 4 | We judge Severe forecasts by **missed days and false alarms together** | Catching more Severe days always costs more false alarms; the balance is a business choice |
| 5 | We keep all **six official categories**, plus a **"Dangerous" alert** (Very Poor or Severe) | Authorities act differently on Very Poor and Severe days; the public needs one clear health warning |
| 6 | We tested a **decision tree** | Best of our models on 2023, and it found the official category limits by itself |
| 7 | We check every model on **five years, not one** | On 2023 alone the first two models looked better than they are: over 2019–2023 neither beats the simple rule |
| 8 | We tested a **random forest** (many trees voting) | **The first model to beat the simple rule in all five years** |
| 9 | We tested **XGBoost** (trees that correct each other) | Also beats the simple rule in all five years; slightly behind the forest overall |
| 10 | We also compare models on **how well they rank risky days** (ROC AUC) | Shows which model's probabilities are best before the final "dial" is set: XGBoost, by a small margin |
| 11 | **XGBoost is the final model**, and the Severe / Dangerous **dial is a setting of the forecast service** | Best at ranking risky days in every year, and a small file; the alert balance can be set and changed later without retraining |
| 12 | **Tree-based models, not logistic regression**, because the data does not follow straight lines | Weather affects pollution in bends and "it depends" patterns, which trees learn by themselves and logistic regression cannot |

> **Where we honestly stand.** Two models now beat "tomorrow = today" **reliably**: the **random forest** (overall score 0.66 against 0.63 on average over five years, ahead in **all five**) and **XGBoost** (0.66 against 0.63, also ahead in all five). The gain is real but modest, about 3 points on the overall score. On the health alert it is clearer: in 2023 the forest warned about **84%** of Dangerous days, against 75% for the simple rule, with the same share of correct warnings. Severe days remain hard. The models catch more of them only by raising more false alarms, so that balance needs a decision (Decision 4).

**How we score a forecast.** Our main score gives **each of the six categories an equal vote**, so rare dangerous days count as much as common ones. (Technical name: macro F1. 1.0 would be perfect.) We also track:
- **Caught:** real Severe days we warned about.
- **Missed:** real Severe days we did not warn about.
- **False alarms:** days we said "Severe" but the air was not that bad.

---

## Decision 1: The benchmark is "tomorrow will be the same as today"

**What we decided.** Any model must beat the simple rule *"tomorrow's air will be in the same category as today's"*.

**Why.** A score means nothing without something to compare it with. We tested four simple forecasts that need no machine learning, and kept the strongest as the benchmark:

| Simple forecast | Right on | Overall score | Severe days caught |
|---|---|---|---|
| Always say "Satisfactory" (the most common category) | 35% of days | 0.09 | 0 of 51 |
| Say what is usual for that city in that month | 53% | 0.39 | 0 of 51 |
| Use the average of the last 7 days | 66% | 0.54 | 8 of 51 |
| **Say tomorrow = today** | **71%** | **0.65** | **27 of 51** |

**What it means.**
- Air pollution has a **strong memory**: a bad day is usually followed by another bad day. That is why "tomorrow = today" is so hard to beat.
- This is also roughly what people do now: dashboards show **today's** air quality, and people plan around it.
- **Its weakness:** it is always **one day late**. It never sees a change coming. The category changes on **29% of days**, and on every one of them this rule is wrong. That is where a model can add value.

---

## Decision 2: Start with a simple, explainable model

**What we decided.** Our first machine-learning model is **logistic regression**: it gives each factor (today's air quality, wind, rain, temperature, season, city…) a weight, and adds them up.

**Why.**
- It is fast and **easy to explain**: we can see which factors push the forecast up or down.
- If a simple model cannot beat the benchmark, the problem lies in the data, not in the choice of model. Better to find that out early.

**What the model learned, and it makes sense:**
- **Today's air quality is by far the strongest factor.**
- **Strong wind, rain and warmer nights lower the chance of a Severe day**: they clear or spread pollution.
- **A sudden jump in pollution tends to fall back the next day**, so a spike alone does not mean tomorrow will be worse.

---

## Decision 3: Don't force the model to predict rare days more often

**What we decided.** We do not apply the common setting that makes rare categories (like Severe) count extra while the model learns. We will instead tune how strongly to favour them.

**Why.** We tested both versions:

| | Model **forced** to favour rare days | Model **not forced** | Benchmark (tomorrow = today) |
|---|---|---|---|
| Overall score | 0.60 | 0.65 | 0.65 |
| Right on | 66% of days | **73%** of days | 71% of days |
| Said "Severe" | 122 times | 31 times | 52 times |
| Severe days caught (of 51) | **37** | 20 | 27 |
| False Severe alarms | 85 | **11** | 25 |

- Forcing the model made it **cry "Severe" far too often**: 122 warnings for 51 real Severe days.
- **The reason:** Severe days have become rarer. They were 3.1% of days in 2015–2022, but only 1.0% in 2023. A model trained to expect them often keeps expecting them.
- Without forcing, the model **ties with the benchmark overall in 2023** and is right on more days (73% against 71%). *Over five years it falls slightly behind (see Decision 7).*

**What it means.** The right setting lies somewhere between the two. We will tune it rather than use either extreme.

*Update: this was true for logistic regression, the decision tree and XGBoost. For the **random forest**, favouring rare days **did** help over five years (Decision 8). This setting has to be checked separately for every type of model.*

---

## Decision 4: Weigh missed Severe days against false alarms

**What we decided.** Severe forecasts are judged on **both** missed days and false alarms. How to balance them is a decision for the people who will use the forecast.

| Out of 51 real Severe days | Forced model | Not forced | Benchmark |
|---|---|---|---|
| Caught | 37 | 20 | 27 |
| Missed | 14 | 31 | 24 |
| False alarms | 85 | 11 | 25 |

**The trade-off in one line:** catching **17 more** Severe days costs **74 more** false alarms, about **4 false alarms for every extra Severe day caught**.

**The question for decision-makers:**

> *Is missing one Severe day worse than raising about four false alarms?*
> - **Yes** (hospitals, school closures, pollution controls): favour catching more Severe days.
> - **No** (a public app, where too many alarms make people stop listening): favour fewer false alarms.

**Good news: most mistakes are small.**
- Of the forced model's 85 false alarms, **76 were Very Poor days**: the air was still hazardous.
- Of the unforced model's 31 missed Severe days, **28 were forecast as Very Poor**: people still got a strong warning.

**What it means.** We can set the balance with a single dial (how sure the model must be before it says "Severe"), once the preferred balance is agreed.

---

## Decision 5: Keep six categories, and add a "Dangerous" alert

**The idea considered.** Merge Very Poor and Severe into one category, "Dangerous", because both harm health, especially for children and older people.

**What we decided.** Keep the **six official categories**, and **add** a "Dangerous" alert on top: *"tomorrow will be Very Poor or Severe"*.

**Why not merge them:**

| | Six categories | After merging into "Dangerous" |
|---|---|---|
| Benchmark score | 0.65 | 0.69 |
| Model score | 0.65 | 0.69 |
| Model vs benchmark | tied | still tied |

- Merging makes **every** score look better, **including the benchmark's**, because the hardest distinction disappears. The model itself does not improve.
- **Authorities act differently** on the two. Under Delhi's Graded Response Action Plan (GRAP), Very Poor triggers Stage II, while Severe triggers Stages III–IV: construction bans, school closures, vehicle restrictions. A single "Dangerous" label could not tell them which stage to prepare for.
- The official health guidance also separates them: Very Poor causes breathing problems on long exposure; Severe affects even healthy people.

**Why add the alert:**

| "Dangerous" alert (419 real Dangerous days in 2023) | Model | Benchmark |
|---|---|---|
| Dangerous days warned about | **81%** | 75% |
| Warnings that were right | **76%** | 75% |

On the alert that matters most for health, **the model beats the benchmark on both counts in 2023**. This is a 2023 result; it has not yet been checked on all five years.

**What the forecast will show:** the exact category (e.g. "Very Poor"), a yes/no **Dangerous** flag, and **how likely** a Dangerous day is (e.g. 83%).

---

## Decision 6: A decision tree, the second model

**What we decided.** Test a **decision tree**: a model that asks a series of yes/no questions (*"is today's AQI above 204?" → "is the weekly average above 178?" → …*) until it reaches a forecast.

**Why.** The simple model gives each factor one fixed weight. A tree can learn **combinations**, such as *"high pollution **and** no wind → Severe"*, and patterns that rise and then fall.

**One important lesson: a tree must be held back.** Left unlimited, it memorised the past (a perfect score on the years it learned from) but did poorly on 2023 (0.51, far below the benchmark). We therefore require every final group of the tree to contain **at least 100 past days**, so it learns general patterns instead of single odd days.

**How it compares in 2023:**

| 2023 | Simple rule (tomorrow = today) | Logistic regression | Decision tree |
|---|---|---|---|
| Overall score | 0.65 | 0.65 | **0.66** |
| Right on | 71% of days | **73%** | 72% |
| Severe days caught (of 51) | **27** | 20 | 25 |
| False Severe alarms | 25 | **11** | 19 |
| Dangerous days warned about | 75% | **81%** | 77% |
| Dangerous warnings that were right | 75% | **76%** | 74% |

**What it means.**
- In 2023 the tree has the best overall score and is better than logistic regression at the **rarest categories** (Good and Severe).
- Logistic regression stays better on the **health alert**.
- **The tree found the official category limits by itself:** its first questions split today's AQI at about 50, 100, 200 and 300, the official CPCB boundaries between categories. Nobody told it those limits.
- Today's air quality does most of the work (83% of the tree's decisions), so the tree behaves like "tomorrow = today, with corrections".

---

## Decision 7: Check every model on five years, not one

**What we decided.** Every model is now checked on **five separate years (2019–2023)**, each time learning only from the years before. A model counts as better only if it beats the simple rule **on average and in most of those years**.

**Why.** All earlier comparisons used **one year**, 2023. But years differ a lot: even the simple rule's score moves between 0.61 and 0.66 from one year to the next. One good or bad year can make a model look better or worse than it really is.

**What we found:**

| Overall score | 2019 | 2020 | 2021 | 2022 | 2023 | **Average** |
|---|---|---|---|---|---|---|
| Simple rule (tomorrow = today) | 0.61 | 0.66 | 0.64 | 0.62 | 0.65 | **0.63** |
| Decision tree | 0.60 | 0.66 | 0.64 | 0.60 | 0.66 | **0.63** |
| Logistic regression | 0.54 | 0.65 | 0.66 | 0.57 | 0.65 | **0.61** |

| | Decision tree | Logistic regression |
|---|---|---|
| Years ahead of the simple rule | 3 of 5, each by a tiny margin | 1 of 5 |
| On average, compared with the simple rule | **a tie** (−0.003) | slightly behind (−0.021) |

**What it means.**
- **2023 made the decision tree look like a winner; over five years it is a tie.** Logistic regression looked like a tie in 2023; over five years it is slightly behind.
- **2022 was hard for every model:** it had far fewer Severe days than the years the models learned from, so the models expected more bad days than happened. Bad-air days have been getting rarer, and the models need to cope with that.
- This check protects against over-promising: **neither of these first two models reliably beats the simple rule.** (The random forest and XGBoost do; see Decisions 8 and 9.)
- It also confirmed the tree's settings: smaller or larger trees did no better.

*Why not simply mix all years together and test on a random sample?* Because neighbouring days are almost identical. A model tested on 2 November after learning from 1 and 3 November can look up the answer instead of forecasting it, which makes it look far better than it will be in real use.

---

## Decision 8: A random forest, the first model to beat the simple rule

**What we decided.** Test a **random forest**: 200 decision trees, each trained on a slightly different sample of past days and allowed to look at only part of the factors at each question. The forecast is their **vote**.

**Why.** A single tree is fragile: small changes in its settings moved its score more than its lead over the simple rule. In a vote, each tree's individual mistakes cancel out, and the patterns they share remain.

**What we found:**

| Overall score | 2019 | 2020 | 2021 | 2022 | 2023 | **Average** |
|---|---|---|---|---|---|---|
| Simple rule (tomorrow = today) | 0.61 | 0.66 | 0.64 | 0.62 | 0.65 | **0.63** |
| **Random forest** | 0.63 | 0.70 | 0.67 | 0.64 | 0.68 | **0.66** |

- **Ahead of the simple rule in all five years**, including 2022, the year every earlier model lost.
- **In 2023 it is the best model on almost every count:** overall score 0.68, right on 73% of days, **84%** of Dangerous days warned about (simple rule 75%), with 76% of warnings right (simple rule 75%).
- **It catches the most Severe days** (31 of 51, against 27 for the simple rule), **but raises the most Severe false alarms** (35). 30 of those 35 were Very Poor days, so the air was still hazardous.
- **The five-year check changed one setting.** On 2023 alone, favouring rare days made no difference. Over five years it was clearly better, so the forest favours them (see the update under Decision 3).
- **The cost:** a large model file (114 MB). It is too big to store in the code repository, so it will be rebuilt by the training script.

---

## Decision 9: XGBoost, the second strong model

**What we decided.** Test **XGBoost**: small trees built **one after another**, each one correcting the mistakes left by the trees before it.

**What we found:**

| | Simple rule | Random forest | XGBoost |
|---|---|---|---|
| Overall score, five-year average | 0.63 | **0.66** (0.664) | 0.66 (0.655) |
| Years ahead of the simple rule | — | **5 of 5** | **5 of 5** |
| 2023 overall score | 0.65 | **0.68** | 0.67 |
| 2023 Severe caught / false alarms (of 51) | 27 / 25 | **31** / 35 | 24 / **18** |
| 2023 Dangerous warned about / right | 75% / 75% | **84%** / 76% | 81% / **77%** |
| Model file | — | 114 MB | **4 MB** |

- XGBoost also beats the simple rule in **every one of the five years**, but it is slightly behind the forest overall, in each year.
- **It is the more cautious model on Severe:** half as many false alarms as the forest, but fewer Severe days caught.
- **One lesson:** XGBoost gets worse if it keeps adding trees (it starts learning noise), so it stops automatically when the score on unseen days stops improving.

---

## Decision 10: A second way to compare models: how well they rank risky days

**What we decided.** Besides the overall score, compare models on **ROC AUC**: *take one day that really was Severe and one that was not; how often does the model give the Severe day the higher risk?* (0.5 = a coin toss, 1.0 = always right.)

**Why.** The overall score judges the model's final category. AUC judges the **risk percentages behind it**, before any "dial" is set (Decision 4). It shows which model gives the best raw material for setting that dial.

**What we found (five-year averages):**

| | Logistic regression | Decision tree | Random forest | XGBoost |
|---|---|---|---|---|
| Ranking of all six categories (AUC) | 0.935 | 0.930 | 0.940 | **0.942** (best in all 5 years) |
| Ranking of Severe days | 0.981 | 0.977 | 0.982 | 0.982 |
| Ranking of Dangerous days | 0.976 | 0.969 | 0.976 | 0.977 |
| Beats the simple rule (overall score) | No | No | **Yes** | **Yes** |

**What it means.**
- **All models rank risky days very well** (above 0.93). The differences between models come mostly from *where the line is drawn*, which is exactly what the Severe dial will tune.
- **XGBoost ranks best, by a small but consistent margin. The forest makes the better final choices today.** Both passed the bar; Decision 11 records the final choice.
- **AUC alone is not enough:** logistic regression ranks well, yet its final forecasts do not beat the simple rule.
- **For Severe days, the models catch more only by raising more false alarms.** At the same number of false alarms as the simple rule, they catch about the same share. For the **Dangerous** alert, the forest and XGBoost catch **82%** of Dangerous days against **75%**, **with the same number of false alarms**.

The detailed points used to pick the best model, and all tuning settings, are in [`model_comparison.md`](model_comparison.md).

---

## Decision 11: XGBoost is the final model; the alert dial lives in the service

**What we decided.**
- **The final model is XGBoost.** It ranks risky days best in every one of the five years, beats the simple rule in all five, and its file is 27 times smaller than the random forest's (4 MB against 114 MB). Its overall score is slightly below the forest's (0.655 against 0.664 on average); we accept that small difference.
- **The "dial" is not built into the model.** The forecast still names the most likely category (the rule that was tested). It also gives **how likely a Severe day is** and **how likely a Dangerous day is**. The level at which a warning is raised becomes a **setting of the forecast service**.

**Why the dial lives in the service.**
- **It can be changed without retraining.** Hospitals, schools and a public app may want different balances between catching Severe days and false alarms (Decision 4). Each can use its own setting.
- **The final test stays honest.** It checks exactly what was validated, and the dial can be agreed afterwards.
- **One rule:** any dial setting must be chosen on 2019–2023, **never on 2025**.

**Final training and test.**
- The model is trained on **2015–2024**: 2024 was never used for any decision, and it adds the most recent year.
- It is tested **once** on **2025** (4,877 city-days).
- The test year is smaller than the 2024–2025 originally kept aside, so the Severe results will be less precise.

**Next steps:**
1. Run the final training and the single test on 2025 (`scripts/train.py`).
2. Build the prediction service: category, chance of Severe, chance of Dangerous, and the alert setting.
3. Package it as a web service in Docker.

**Already done:** tuning the simple model's strength setting (regularisation) confirmed that the default setting is the best one; four model types have been tuned and compared on five years.

---

## Decision 12: Tree-based models over logistic regression, because the data is not a straight line

**What we decided.** Use a tree-based model (random forest or XGBoost) for the final forecast, not logistic regression.

**Why.** Logistic regression works like a ruler: it gives each factor **one fixed effect**, always the same size and in the same direction. We checked whether our data behaves like that. It mostly does not.

**What we found** (training years 2015–2022, average AQI on the next day):

| Factor | What the data shows | Why a straight line fails |
|---|---|---|
| Temperature | Worst on cold days (361 below 10 °C), best at 25–30 °C (108), **worse again** above 35 °C (210, summer dust) | The effect turns around |
| Wind | **No effect below 2 m/s** (about 200), then a steep improvement (73 at 5–8 m/s) | The effect starts only after a threshold |
| Rain | The first few mm clean the air most; heavier rain adds little | The effect flattens |
| Month | August averages 76, December 254, yet the correlation is **almost zero** | The year goes down and back up, so a straight line sees "no effect" |
| Wind × season | Delhi, Nov–Jan: windy days **about 90 points cleaner** (276 vs 368). Delhi, Jun–Sep: **no difference** (130 vs 130) | The effect of one factor depends on another |

Only **today's AQI** behaves like a straight line.

**What it means.**
- **Logistic regression only copes if we reshape factors by hand** (month as a position on a circle, rain on a log scale), and it still cannot learn "wind matters only in winter".
- **Trees learn bends and "it depends" patterns by themselves.** They ask questions in sequence: *is it November?* Then *is the wind below 2 m/s?*
- **The results confirm it.** Over five years, the random forest and XGBoost beat the simple rule in **5 of 5** years (average 0.66). Logistic regression beat it in **1 of 5** (0.61).

**In one sentence:** pollution does not respond to the weather in a straight line, so we chose models that learn those patterns from the data.

---

## Appendix: technical notes for the team

*Not needed for the management discussion; kept here so the details are not lost.*

**Evaluation**
- Main score: **macro F1** (average of the six per-class F1 scores). Also: accuracy, Severe recall (caught ÷ real) and precision (right ÷ said), change days vs unchanged days.
- The benchmark is **persistence**. Scores in the tables above are rounded; exact values: persistence macro F1 0.651; logistic regression with `class_weight='balanced'` 0.598, with no class weights 0.647.

**Model setup (`experiments/lr_train_baseline.ipynb`)**
- Pipeline: `DictVectorizer(sparse=False)` → `StandardScaler()` → `LogisticRegression(max_iter=3000)`, fitted on 2015–2022 only.
- DictVectorizer one-hot encodes `city` and takes one dictionary per row, the same format as the web service's JSON input. StandardScaler is needed because `aqi_today` (0–500) and the 0/1 flags are on very different scales.
- Features: 15 numeric + 7 yes/no + `city`; `aqi_roll7_mean` left out (it nearly duplicates `aqi_today`).
- Pitfalls: the model stores classes alphabetically, so probabilities must be paired with `model.classes_`; a `ConvergenceWarning` is raised as an error; the saved bundle includes the city pressure normals and the city list, and the service must reject unknown cities.

**Class weights**
- `'balanced'` gives Severe about 7× weight and lowers the bar for predicting it: the Severe intercept is −5.25 with weights against −7.01 without. The `aqi_today` weight is 6.87 vs 7.21, so `aqi_today` is not the cause of the over-prediction.
- Predicted ÷ actual share per class (1.0 = perfect). No weights: Good 0.51, Satisfactory 1.06, Moderate 1.02, Poor 0.98, Very Poor 1.12, Severe 0.61. Balanced: Good 1.78, Satisfactory 0.82, Moderate 0.91, Poor 1.20, Very Poor 1.09, Severe 2.39.

**Misses vs false alarms**
- The preferred tool is a **decision threshold on P(Severe)**, rather than class weights, because it changes only the Severe decision.

**Regularisation (`C`)**
- Tuned over 0.0001–100 for both weightings (`lr_train_baseline.ipynb`). Validation macro F1 rises until `C = 1` and then stays flat, so `C = 1` (the default) is kept. Stronger L2 underfits (both training and validation scores fall) and, with balanced weights, *increases* Severe calls (122 → 440). The model does not overfit: training 0.649 vs validation 0.647.
- Saved as `experiments/models/lr_best.bin` (no class weights, `C = 1`).

**Decision tree (`dt_train.ipynb`)**
- `DecisionTreeClassifier` (CART, Gini), `DictVectorizer` for `city`, raw features (no transforms or scaling), all four trend features.
- Unlimited tree: 6,700 leaves, training macro F1 1.000, 2023 0.513 (overfitting).
- 60 settings tuned (depth 3–unlimited × min_samples_leaf 1–400 × class weights none/balanced). Best: `max_depth=10`, `min_samples_leaf=100`, no class weights, 204 leaves, 2023 macro F1 0.658 (training 0.668). Balanced weights best 0.636.
- Feature importance: `aqi_today` 0.83, then `aqi_roll7_mean`, `precip_sum`, `wind_speed_max`. First splits on `aqi_today` at 49.5, 103.5, 204.5, 307.5.
- Saved as `experiments/models/dt_best.bin` when the notebook is run.

**Rolling validation (`utility.rolling_validation`)**
- Rounds: learn from years before Y, score Y, for Y = 2019–2023. Refuses to score 2024 or later. Data-derived values (city pressure normals) are recomputed per round from that round's training years.
- Macro F1 per year (2019 / 2020 / 2021 / 2022 / 2023, mean, mean vs persistence, years ahead):
  - persistence: 0.609 / 0.655 / 0.638 / 0.619 / 0.651, mean 0.634
  - tree (depth 10, leaf 100): 0.604 / 0.657 / 0.643 / 0.596 / 0.658, mean 0.631, −0.003, 3 of 5
  - logistic regression (no weights, C = 1): 0.543 / 0.647 / 0.655 / 0.574 / 0.647, mean 0.613, −0.021, 1 of 5
  - other tree settings (leaf 50, 200, 400; depth 5): mean 0.623–0.631, never better than the chosen one.
- A random split was rejected: in a random mix of 2015–2023, 84% of scored days have the next day of the same city in training (whose `aqi_today` is the answer), and the unlimited tree's gap to persistence shrank from −0.138 to −0.060, i.e. the random split flatters memorising models.

**Random forest (`rf_train.ipynb`)**
- `RandomForestClassifier(n_estimators=200, max_features=0.5, min_samples_leaf=5, class_weight='balanced_subsample', random_state=1)`; same raw features as the tree.
- 200 unlimited trees (leaf 1): training 1.000, 2023 0.670 (a single unlimited tree: 0.513). Tree count curve flat from about 100 trees on.
- 32 settings on 2023 (`max_features` sqrt/0.33/0.5/1.0 × leaf 1/5/20/50 × weights). Top 10 within 0.005, so the 3 best per weighting went to rolling validation. Chosen: mean 0.664 (+0.030, 5 of 5 years, spread 0.027, training gap 0.230). Best unweighted forest: 0.652.
- 2023: macro F1 0.679, accuracy 0.734, Severe 31 / 20 / 35, Dangerous 0.840 / 0.760. Importance: `aqi_today` 0.37, `aqi_roll7_mean` 0.17, `temp_range` 0.05.
- File 114 MB, above GitHub's 100 MB limit: `experiments/models/*.bin` should not be committed.

**XGBoost (`xgb_train.ipynb`)**
- `CategoryXGB` wrapper in `utility.py` (XGBoost needs the classes as 0–5). `learning_rate=0.05`, `max_depth=3`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.8`, 628 rounds from early stopping (patience 50, 2023 log loss), i.e. 3,768 trees (one per category per round), no class weights. `min_child_weight` limits the sum of 2·p·(1−p) per leaf, not the number of days.
- 12 settings on 2023 (depth 3/5/7 × `min_child_weight` 1/10 × weights); 6 to rolling validation. Depth 7 (0.656, 4 of 5 years) and depth 3 (0.655, 5 of 5) tied; the tie-break chose depth 3 (more years ahead, training gap 0.086).
- 2023: macro F1 0.668, accuracy 0.735, Severe 24 / 27 / 18, Dangerous 0.809 / 0.767. File 4.2 MB. On macOS it needs `brew install libomp`.

**ROC AUC (`model_comparison.ipynb`, `utility.auc_scores`)**
- One-vs-rest AUC per class, macro = mean of the six, Dangerous = AUC of P(Very Poor) + P(Severe). Probabilities are reordered by `model.classes_` (`utility.proba_table`). Persistence uses 0/1 "probabilities" (one ROC point), so its AUC is not comparable with the models'.
- Five-year mean macro AUC: logistic regression 0.935, tree 0.930, forest 0.940, XGBoost 0.942 (best in every year); persistence 0.782. Severe: 0.981 / 0.977 / 0.982 / 0.982. Dangerous: 0.976 / 0.969 / 0.976 / 0.977.
- Class weights barely change AUC (logistic regression Severe AUC 0.976 balanced vs 0.975 without), but change macro F1 a lot: they move the threshold, not the ranking.
- At persistence's false-alarm rate on 2023: Severe 45–57% caught (persistence 53%); Dangerous 82% for forest and XGBoost (persistence 75%).

**"Dangerous" alert**
- Derived from the six-class prediction: `dangerous = category in ['Very Poor', 'Severe']`; `p_dangerous = P(Very Poor) + P(Severe)`.
- Tested alternatives (no class weights): six-class model merged after prediction gives Dangerous recall 0.807 / precision 0.761; a model trained on five classes gives 0.821 / 0.756, essentially the same. Persistence: 0.749 / 0.753. Five-class macro F1: persistence 0.693, models 0.691–0.692.

**Non-linearity checks (Decision 12)**
- Data: 2015–2022 only (32,714 city-days). Target: tomorrow's AQI, the next calendar day's `aqi_today` of the same city (rows without a next day dropped).
- Method 1, binned means: average tomorrow's AQI per range of a feature. Even steps = linear; bends, thresholds or a U shape = non-linear.
- Method 2, Pearson (straight-line fit) vs Spearman (any steady up or down). A large gap means a curve; both near 0 with a clear binned pattern means the effect turns around.
  - `aqi_today` +0.91 / +0.91, `wind_speed_mean` −0.35 / −0.35, `temp_mean` −0.48 / −0.35, `humidity_mean` −0.29 / −0.45, `precip_sum` −0.29 / −0.62, `month` +0.04 / −0.01.
  - Wind has no Pearson/Spearman gap, yet its binned means are flat up to 2 m/s: correlation alone can miss a threshold.
- Method 3, interactions: binned means split by a second factor (wind: calm < 1.5 m/s, windy > 3 m/s; Nov–Jan vs Jun–Sep).
- Method 4, model comparison on the same data: rolling macro F1 logistic regression 0.613 (1 of 5 years) vs random forest 0.664 and XGBoost 0.655 (5 of 5).
