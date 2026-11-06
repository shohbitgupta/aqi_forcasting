# Modelling decisions

Decision log for the next-day AQI category model. Full results and tuning grids: [`model_comparison.md`](model_comparison.md); accuracy experiments: [`accuracy_experiments.md`](accuracy_experiments.md); feature transforms: [`feature_transformation_notes.md`](feature_transformation_notes.md).

**Conventions.** The main metric is macro F1. Single-year results are on 2023 (5,104 city-days, 51 Severe), a year no model trained on. "Five years" means rolling validation over 2019–2023: each year is scored by a model trained only on the years before it. The final test year, 2025, was used once, after every decision below.

| # | Decision | Key evidence |
|---|---|---|
| 1 | Persistence ("tomorrow = today") is the baseline | Best of four non-ML forecasts: macro F1 0.651, accuracy 71% on 2023 |
| 2 | Start with logistic regression | Explainable weights; sets the bar for more complex models |
| 3 | No class weights, except for the random forest | Balanced weights: 122 Severe calls for 51 Severe days, macro F1 0.598 vs 0.647 |
| 4 | Trade misses against false alarms with a probability threshold | ~4 extra false alarms per extra Severe day caught |
| 5 | Keep six categories; add a Dangerous alert (Very Poor or Severe) | Merging inflates the baseline equally; GRAP acts on each category separately |
| 6 | Decision tree: depth 10, at least 100 days per leaf | Unlimited tree: 1.000 on training, 0.513 on 2023 |
| 7 | Every model is checked on five years; no random split | 2023 alone made the tree look like a winner; over five years it ties persistence |
| 8 | Random forest: the first model ahead in all five years | Macro F1 0.664 vs 0.634, 5 of 5 years |
| 9 | XGBoost: also ahead in all five years | Macro F1 0.655, 5 of 5 years, 4 MB |
| 10 | Also compare models on ranking (ROC AUC) | XGBoost best macro AUC in every year (0.942 mean) |
| 11 | XGBoost is the final model; alert thresholds are service settings | Best ranking, 5 of 5 years, 27× smaller than the forest |
| 12 | Tree models over logistic regression | Weather effects bend, plateau and interact; only today's AQI is linear |

---

## 1. Persistence is the baseline

**Decision.** A model must beat persistence: tomorrow's category equals today's.

**Evidence** (2023, `experiments/baseline_forecast.ipynb`):

| Forecast | Accuracy | Macro F1 | Severe caught (of 51) |
|---|---|---|---|
| Always "Satisfactory" | 35% | 0.09 | 0 |
| City-month climatology | 53% | 0.39 | 0 |
| Mean of the last 7 days | 66% | 0.54 | 8 |
| **Persistence** | **71%** | **0.65** | **27** |

AQI is strongly autocorrelated, which makes persistence hard to beat. It is wrong on every day the category changes (29% of days), which is where a model can add value.

## 2. Logistic regression first

**Decision.** The first model is logistic regression: `DictVectorizer` → `StandardScaler` → `LogisticRegression`, trained on 2015–2022 (`experiments/lr_train_baseline.ipynb`).

**Evidence.** The learned weights are physically sensible: today's AQI dominates; wind, rain and warmer nights lower the chance of Severe; a sudden jump tends to revert the next day. `C` was tuned over 0.0001–100: validation macro F1 rises until `C = 1` and is flat after it, so the default is kept. No overfitting (training 0.649, validation 0.647). Inputs: 15 numeric, 7 binary and `city`; `aqi_roll7_mean` is left out (it nearly duplicates `aqi_today`).

## 3. No class weights, except for the random forest

**Decision.** Train without class weights. Rare-class handling is done with a threshold (Decision 4) instead.

**Evidence** (logistic regression, 2023):

| | Balanced weights | No weights | Persistence |
|---|---|---|---|
| Macro F1 | 0.598 | **0.647** | 0.651 |
| Accuracy | 66% | **73%** | 71% |
| Severe forecasts | 122 | 31 | 52 |
| Severe caught / false alarms | 37 / 85 | 20 / **11** | 27 / 25 |

Severe was 3.1% of training days (2015–2022) but 1.0% of 2023, so a model weighted towards it over-predicts it. Class weights move the decision threshold, not the ranking: Severe ROC AUC is 0.976 with weights and 0.975 without. The same held for the decision tree and XGBoost. The random forest is the exception: `balanced_subsample` was better over five years (Decision 8), so this setting is checked per model type.

## 4. Misses vs false alarms: a threshold, not retraining

**Decision.** Severe forecasts are judged on misses and false alarms together. The balance between them is set by a threshold on the predicted probability, chosen by the user of the forecast.

**Evidence** (2023, of 51 Severe days): balanced weights caught 17 more Severe days than no weights, at the cost of 74 more false alarms (about 4 per extra day caught). Most errors are one category away: 76 of the 85 false alarms were Very Poor days, and 28 of the 31 missed Severe days were forecast Very Poor.

## 5. Six categories plus a Dangerous alert

**Decision.** Keep the six CPCB categories. Add a Dangerous alert: `p_dangerous = P(Very Poor) + P(Severe)`.

**Evidence.**
- Merging Very Poor and Severe raises macro F1 for both the model and persistence (0.65 → 0.69) without changing the gap between them.
- Delhi's GRAP acts differently on the two: Very Poor triggers Stage II; Severe triggers Stages III–IV (construction bans, school closures, vehicle restrictions).
- Dangerous alert on 2023 (419 Dangerous days): logistic regression warned on 81% with 76% precision; persistence 75% and 75%.
- A model trained on five classes gives the same alert as merging the six-class output: 0.821 / 0.756 vs 0.807 / 0.761.

## 6. Decision tree, regularised

**Decision.** `DecisionTreeClassifier` with `max_depth=10`, `min_samples_leaf=100`, no class weights, on raw features (`experiments/dt_train.ipynb`).

**Evidence.**
- An unlimited tree (6,700 leaves) scored 1.000 on training and 0.513 on 2023.
- Best of 60 settings: 2023 macro F1 0.658 (training 0.668), 204 leaves.
- Its first splits on `aqi_today` fall at 49.5, 103.5, 204.5 and 307.5, close to the CPCB category boundaries, without being told them.
- `aqi_today` carries 0.83 of the feature importance: the tree behaves like persistence with corrections.

## 7. Rolling validation over five years

**Decision.** Every model is scored on 2019–2023, each year trained only on the years before it. A model counts as better only if it beats persistence on average and in most years.

**Evidence.** Persistence alone varies from 0.61 to 0.66 between years, so one year is not enough.

| Macro F1 | 2019 | 2020 | 2021 | 2022 | 2023 | Mean | Years ahead |
|---|---|---|---|---|---|---|---|
| Persistence | 0.609 | 0.655 | 0.638 | 0.619 | 0.651 | 0.634 | – |
| Decision tree | 0.604 | 0.657 | 0.643 | 0.596 | 0.658 | 0.631 | 3 of 5 |
| Logistic regression | 0.543 | 0.647 | 0.655 | 0.574 | 0.647 | 0.613 | 1 of 5 |

A random split was rejected: in a random mix of 2015–2023, 84% of scored days have the same city's next day in training, whose `aqi_today` is the answer. It flatters memorising models: the unlimited tree's gap to persistence shrank from −0.138 to −0.060.

## 8. Random forest

**Decision.** `RandomForestClassifier(n_estimators=200, max_features=0.5, min_samples_leaf=5, class_weight='balanced_subsample')` (`experiments/rf_train.ipynb`).

**Evidence.**
- Five-year macro F1 0.664 (+0.030 over persistence), ahead in 5 of 5 years, including 2022, which every earlier model lost.
- 2023: macro F1 0.679, accuracy 73.4%, Dangerous recall 84% at 76% precision.
- Catches the most Severe days (31 of 51) but raises the most Severe false alarms (35, of which 30 were Very Poor).
- Model file: 114 MB.

## 9. XGBoost

**Decision.** `CategoryXGB` (XGBoost with category names, `scripts/utility.py`): `max_depth=3`, `min_child_weight=10`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, no class weights, 628 rounds from early stopping on 2023 log loss (patience 50) (`experiments/xgb_train.ipynb`).

**Evidence.**
- Five-year macro F1 0.655, ahead of persistence in 5 of 5 years.
- Depth 7 tied on macro F1 (0.656) but was ahead in only 4 of 5 years; depth 3 was chosen.
- 2023: macro F1 0.668, Severe 24 caught with 18 false alarms (the forest: 31 and 35), Dangerous 81% at 77% precision.
- Model file: 4.2 MB.

## 10. Ranking quality (ROC AUC)

**Decision.** Compare models also on ROC AUC: how well the predicted probabilities rank risky days, independent of the threshold (`experiments/model_comparison.ipynb`).

**Evidence** (five-year means):

| | Logistic regression | Decision tree | Random forest | XGBoost |
|---|---|---|---|---|
| Macro AUC | 0.935 | 0.930 | 0.940 | **0.942** (best in all 5 years) |
| Severe AUC | 0.981 | 0.977 | 0.982 | 0.982 |
| Dangerous AUC | 0.976 | 0.969 | 0.976 | 0.977 |
| Beats persistence on macro F1 | No | No | Yes | Yes |

AUC alone is not sufficient: logistic regression ranks well but does not beat persistence. At persistence's false-alarm rate, the forest and XGBoost catch 82% of Dangerous days against 75%. The final test also reports PR AUC, the stricter ranking measure for rare classes.

## 11. Final model: XGBoost; thresholds in the service

**Decision.**
- XGBoost is the final model. Its macro F1 is slightly below the forest's (0.655 vs 0.664), which is accepted: it ranks risky days best in every year, beats persistence in all five, and is 27 times smaller (4 MB vs 114 MB).
- The model returns the most likely category plus P(Severe) and P(Dangerous). Alert thresholds are request parameters of the service (`severe_threshold`, `dangerous_threshold`), so they can change without retraining. Any default threshold must be chosen on 2019–2023, never on 2025.
- Final training on 2015–2024; a single test on 2025 (4,877 city-days).

**Result** (2025, `scripts/train.py`): macro F1 0.671 vs 0.659 for persistence; Dangerous recall 82.7% at 77.6% precision; Dangerous PR AUC 0.892 vs 0.610.

## 12. Tree models over logistic regression

**Decision.** Use a tree ensemble for the final model, not logistic regression.

**Evidence** (2015–2022, mean next-day AQI by feature range):

| Feature | Pattern | Why a linear model fails |
|---|---|---|
| Temperature | 361 below 10 °C, 108 at 25–30 °C, 210 above 35 °C (summer dust) | U-shaped |
| Wind | ~200 below 2 m/s, 73 at 5–8 m/s | No effect until a threshold |
| Rain | The first few mm clean the air most | Saturates |
| Month | August 76, December 254, yet correlation ≈ 0 | Cyclical |
| Wind × season (Delhi) | Nov–Jan: windy days ~90 points cleaner (276 vs 368); Jun–Sep: no difference (130 vs 130) | Interaction |

Only today's AQI is close to linear (Pearson and Spearman both 0.91). Hand-made transforms (sin/cos month, log rain) fix some shapes but not interactions. Over five years, the random forest and XGBoost beat persistence in 5 of 5 years (0.664, 0.655); logistic regression in 1 of 5 (0.613).
