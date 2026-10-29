# Model comparison: four models against "tomorrow = today"

**Task:** forecast tomorrow's air quality category for 14 Indian cities. There are six categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe), and every city-day has exactly **one** of them. So this is a **multi-class** classification problem: one label out of six per row. It is not **multi-label**, which would allow several labels on the same row at once.

**The four models:**

| Model                             | How it forecasts                                                                                  | Notebook                  | Saved model                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------- | --------------------------------- |
| Logistic regression (multinomial) | Gives every factor one weight, adds them up, turns the sums into six probabilities                | `lr_train_baseline.ipynb` | `experiments/models/lr_best.bin`  |
| Decision tree                     | A chain of yes/no questions ("is today's AQI above 204?") ending in a forecast                    | `dt_train.ipynb`          | `experiments/models/dt_best.bin`  |
| Random forest                     | 200 trees, each trained on a different sample of days and features, vote                          | `rf_train.ipynb`          | `experiments/models/rf_best.bin`  |
| Gradient boosted trees (XGBoost)  | 628 rounds of small trees built one after another (one tree per category per round, 3,768 trees), each round correcting the mistakes left by the rounds before it | `xgb_train.ipynb`         | `experiments/models/xgb_best.bin` |

All four are compared with the benchmark **persistence** (_"tomorrow's category = today's"_). AUC numbers come from `model_comparison.ipynb`.

**How the models are tested:**

- **One year:** train on 2015–2022, score on **2023** (5,104 city-days, 51 Severe, 419 Dangerous).
- **Five years** (rolling validation): for each year from 2019 to 2023, train on all earlier years and score that year.
- 2024–2025 was **locked** and not used for any of the numbers below. For the final model, 2024 joins the training years and **2025** is the single final test (section 3.5).

**Contents**

1. [Results on 2023](#1-results-on-2023)
2. [Results over five years](#2-results-over-five-years-20192023)
3. [Choosing the best model with ROC AUC](#3-choosing-the-best-model-with-roc-auc)
4. [Tuning parameters of each model](#4-tuning-parameters-of-each-model)
5. [Strengths and weaknesses at a glance](#5-strengths-and-weaknesses-at-a-glance)

---

## 1. Results on 2023

### 1.1 Main scores

| 2023                                         | Persistence | Logistic regression | Decision tree | Random forest | XGBoost   |
| -------------------------------------------- | ----------- | ------------------- | ------------- | ------------- | --------- |
| **Macro F1** (main score)                    | 0.651       | 0.647               | 0.658         | **0.679**     | 0.668     |
| Accuracy                                     | 0.710       | 0.729               | 0.723         | 0.734         | **0.735** |
| Severe caught (of 51)                        | 27          | 20                  | 25            | **31**        | 24        |
| Severe missed                                | 24          | 31                  | 26            | **20**        | 27        |
| Severe false alarms                          | 25          | **11**              | 19            | 35            | 18        |
| Dangerous recall (warned ÷ real)             | 0.749       | 0.807               | 0.768         | **0.840**     | 0.809     |
| Dangerous precision (right ÷ warned)         | 0.753       | 0.761               | 0.740         | 0.760         | **0.767** |
| Change days right (category moves overnight) | 0%          | **35.3%**           | 27.8%         | 26.8%         | 28.0%     |
| Unchanged days right                         | **100%**    | 88.3%               | 90.4%         | 92.4%         | 92.1%     |
| Macro F1 on the training years (2015–2022)   | —           | 0.649               | 0.668         | 0.900         | 0.733     |

### 1.2 F1 per category

| F1, 2023     | Persistence | Logistic regression | Decision tree | Random forest | XGBoost  |
| ------------ | ----------- | ------------------- | ------------- | ------------- | -------- |
| Good         | 0.62        | 0.52                | 0.62          | **0.66**      | 0.61     |
| Satisfactory | 0.79        | 0.80                | 0.80          | 0.80          | **0.81** |
| Moderate     | 0.72        | **0.74**            | 0.73          | **0.74**      | **0.74** |
| Poor         | 0.59        | 0.61                | 0.59          | **0.63**      | 0.61     |
| Very Poor    | 0.67        | **0.72**            | 0.68          | 0.71          | **0.72** |
| Severe       | 0.52        | 0.49                | **0.53**      | **0.53**      | 0.52     |

### 1.3 ROC AUC per category

AUC is computed one category against the rest. `Dangerous` uses P(Very Poor) + P(Severe).

| AUC, 2023                      | Persistence\* | Logistic regression | Decision tree | Random forest | XGBoost   |
| ------------------------------ | ------------- | ------------------- | ------------- | ------------- | --------- |
| Good                           | 0.801         | 0.956               | 0.955         | 0.963         | **0.966** |
| Satisfactory                   | 0.835         | 0.924               | 0.928         | **0.935**     | **0.935** |
| Moderate                       | 0.779         | 0.885               | 0.882         | 0.891         | **0.893** |
| Poor                           | 0.756         | 0.917               | 0.913         | 0.918         | **0.920** |
| Very Poor                      | 0.822         | **0.975**           | 0.966         | 0.972         | **0.975** |
| Severe                         | 0.762         | 0.975               | **0.986**     | 0.976         | 0.979     |
| **Macro** (average of the six) | 0.793         | 0.939               | 0.938         | 0.943         | **0.945** |
| **Dangerous**                  | 0.864         | **0.982**           | 0.974         | 0.980         | **0.982** |

\* Persistence has no probabilities (100% for today's category, 0% for the rest), so it gets one point on the ROC curve, not a curve. Its AUC is shown for reference only; it cannot be compared fairly with the models'.

---

## 2. Results over five years (2019–2023)

### 2.1 Macro F1 per year

| Macro F1            | 2019  | 2020  | 2021  | 2022  | 2023  | **Average** |
| ------------------- | ----- | ----- | ----- | ----- | ----- | ----------- |
| Persistence         | 0.609 | 0.655 | 0.638 | 0.619 | 0.651 | **0.634**   |
| Logistic regression | 0.543 | 0.647 | 0.655 | 0.574 | 0.647 | 0.613       |
| Decision tree       | 0.604 | 0.657 | 0.643 | 0.596 | 0.658 | 0.631       |
| Random forest       | 0.633 | 0.695 | 0.674 | 0.639 | 0.679 | **0.664**   |
| XGBoost             | 0.631 | 0.684 | 0.673 | 0.621 | 0.668 | 0.655       |

### 2.2 Macro ROC AUC per year

| Macro AUC           | 2019      | 2020      | 2021      | 2022      | 2023      | **Average** |
| ------------------- | --------- | --------- | --------- | --------- | --------- | ----------- |
| Persistence\*       | 0.765     | 0.794     | 0.785     | 0.776     | 0.793     | 0.782       |
| Logistic regression | 0.924     | 0.938     | 0.938     | 0.936     | 0.939     | 0.935       |
| Decision tree       | 0.916     | 0.935     | 0.931     | 0.930     | 0.938     | 0.930       |
| Random forest       | 0.926     | 0.947     | 0.940     | 0.943     | 0.943     | 0.940       |
| XGBoost             | **0.928** | **0.948** | **0.943** | **0.945** | **0.945** | **0.942**   |

### 2.3 AUC for the categories that matter most

| Five-year average | Persistence\* | Logistic regression | Decision tree | Random forest | XGBoost |
| ----------------- | ------------- | ------------------- | ------------- | ------------- | ------- |
| Severe AUC        | 0.753         | 0.981               | 0.977         | 0.982         | 0.982   |
| Dangerous AUC     | 0.851         | 0.976               | 0.969         | 0.976         | 0.977   |

Best Severe AUC per year: logistic regression (2019), XGBoost (2020, 2022), random forest (2021), decision tree (2023). On Severe and Dangerous, **no model leads consistently**: the top three are within 0.001 on average.

### 2.4 Five-year summary

|                                            | Logistic regression | Decision tree       | Random forest        | XGBoost              |
| ------------------------------------------ | ------------------- | ------------------- | -------------------- | -------------------- |
| Mean macro F1                              | 0.613               | 0.631               | **0.664**            | 0.655                |
| vs persistence (0.634)                     | −0.021              | −0.003              | **+0.030**           | +0.021               |
| Years ahead of persistence                 | 1 of 5              | 3 of 5              | **5 of 5**           | **5 of 5**           |
| **Passes the bar**†                        | No                  | No                  | **Yes**              | **Yes**              |
| Macro F1 spread between years (std)        | 0.051               | 0.030               | **0.027**            | 0.028                |
| Training gap (training F1 − validation F1) | **0.026**           | **0.026**           | 0.230                | 0.086                |
| Mean macro AUC                             | 0.935               | 0.930               | 0.940                | **0.942**            |
| Years with the best macro AUC              | 0                   | 0                   | 0                    | **5**                |
| Training time on 2015–2022                 | 0.8 s               | **0.6 s**           | 4.6 s                | 6.9 s                |
| Model file size                            | **7 KB**            | 54 KB               | 114 MB               | 4.2 MB               |
| Easy to explain                            | **Yes (weights)**   | **Yes (questions)** | Partly (importances) | Partly (importances) |

† **The bar:** beat persistence on the five-year average of macro F1 **and** in most of the five years.

---

## 3. Choosing the best model with ROC AUC

### 3.1 What AUC tells us, and what it does not

- **What it measures:** ranking. Take one day that really was Severe and one that was not. AUC is the chance the model gave the Severe day the higher Severe probability. 0.5 is a coin toss; 1.0 is a perfect ranking.
- **Its strength:** it does **not** depend on a threshold. It judges the probabilities **before** the model turns them into one category. So it shows how good a model _can_ be once its thresholds are tuned.
- **Its blind spot:** a model can rank days well and still pick the wrong category in the end. **Logistic regression is the proof.** Its five-year AUC (0.935) is higher than the decision tree's (0.930), yet its macro F1 is the lowest (0.613) and below persistence.
- **Class weights move the bar, not the ranking.** Removing them from logistic regression changed its 2023 macro F1 from 0.598 to 0.647, while its Severe AUC stayed at 0.975.

So AUC is used **together with** macro F1, never alone.

### 3.2 The points to check, in order

| #   | Point                                                          | Why                                                                                               | Question asked                                                             |
| --- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1   | **Pass the bar on macro F1 first**                             | A good ranking is worthless if the final forecasts are not better than "tomorrow = today"         | Does it beat persistence on the five-year average and in most years?       |
| 2   | **Compare macro AUC over five years, not one**                 | One year is noisy: the 2023-only order of models was not always the five-year order               | Which model has the highest five-year mean macro AUC?                      |
| 3   | **Check consistency**                                          | A higher average can come from one lucky year                                                     | In how many of the five years is it the best?                              |
| 4   | **Check the categories that matter**: Severe and Dangerous AUC | The health warning matters more than telling Satisfactory from Moderate                           | Is it also best (or tied) on Severe and Dangerous?                         |
| 5   | **Judge the size of the difference**                           | A gap smaller than the year-to-year movement can be luck                                          | Is the gap consistent across years, or within the noise?                   |
| 6   | **Look at the left part of the ROC curve**, not only the area  | Only low false-alarm rates are usable in practice; the area also counts settings nobody would use | At an acceptable number of false alarms, how many real days does it catch? |
| 7   | **Do not compare with persistence's AUC**                      | Persistence has one point, not a curve, so its AUC is not comparable                              | (Use macro F1 and the alert scores to compare with persistence)            |
| 8   | **Practical tie-breakers**                                     | When the scores are close, cost and trust decide                                                  | File size, speed, ease of explaining                                       |

### 3.3 Applying the points

| #   | Point                               | Logistic regression        | Decision tree         | Random forest                                         | XGBoost                                               | Leader                                                             |
| --- | ----------------------------------- | -------------------------- | --------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------ |
| 1   | Passes the bar                      | No (−0.021, 1 of 5)        | No (−0.003, 3 of 5)   | **Yes** (+0.030, 5 of 5)                              | **Yes** (+0.021, 5 of 5)                              | Forest, XGBoost                                                    |
| 2   | Five-year macro AUC                 | 0.935                      | 0.930                 | 0.940                                                 | **0.942**                                             | XGBoost                                                            |
| 3   | Years with the best macro AUC       | 0                          | 0                     | 0                                                     | **5 of 5**                                            | XGBoost                                                            |
| 4   | Severe / Dangerous AUC (five years) | 0.981 / 0.976              | 0.977 / 0.969         | 0.982 / 0.976                                         | 0.982 / 0.977                                         | Tie (forest, XGBoost, logistic regression)                         |
| 5   | Size of the macro AUC gap           |                            |                       | −0.002 vs XGBoost                                     |                                                       | Small (0.0017) but **the same in every year** → a real, small edge |
| 6   | Left part of the ROC curve (2023)   | Dangerous: 80% caught      | Dangerous: 73% caught | Severe: 67% caught at 1% false alarms; Dangerous: 82% | Severe: 69% caught at 1% false alarms; Dangerous: 82% | Tie (forest, XGBoost)                                              |
| 7   | Persistence's AUC (0.782)           | not compared               | not compared          | not compared                                          | not compared                                          | — (persistence is compared on macro F1, point 1)                   |
| 8   | Tie-breakers                        | Smallest, most explainable | Small, explainable    | 114 MB file                                           | 4.2 MB file                                           | XGBoost (between the two finalists)                                |

The "left part of the ROC curve" row compares models at the **same** false-alarm rate as persistence: 103 false alarms in 2023, or 2.2% of the other days. At that rate, persistence caught 75% of the Dangerous days.

**For Severe, there is no gain at persistence's own false-alarm level.** Persistence raised 25 false alarms (0.5% of the other days) and caught 53% of the Severe days. At that same rate, the models catch 45–57%. The models only catch more Severe days if more false alarms are accepted: at about 50 false alarms (1%), the forest catches 67% and XGBoost 69%.

### 3.4 Verdict

- **By ROC AUC, the best model is XGBoost.** Of the two models that pass the bar, it has the highest five-year macro AUC (0.942), it is the best in **all five years**, and it ties for the best Severe and Dangerous AUC. Its file is 27× smaller than the forest's.
- **By macro F1 at the default decision** (pick the most likely category), **the random forest is better**: 0.664 against 0.655 over five years, and ahead of XGBoost in each year. Its balanced class weights move its decisions towards the rare categories.
- **These two views fit together.** XGBoost has slightly better probabilities; the forest currently turns its probabilities into categories slightly better. The planned **threshold step** (deciding how sure a model must be before it says "Severe" or "Very Poor") changes exactly that second part.

**Recommendation at this stage** (replaced by the final decision in section 3.5):

1. Take **both XGBoost and the random forest** to the threshold step.
2. Pick the final model by its **five-year macro F1 after thresholds**, with the same bar as before.
3. **If they tie** (within 0.002), pick **XGBoost**: better ranking in every year, a smaller file, faster to load.
4. Until then, the **random forest is the provisional leader**, because it is the best at the decision that is actually made today.

Logistic regression and the decision tree are **not candidates**: neither passes the bar. Logistic regression's very good Severe AUC (0.981) shows its Severe probabilities are useful, but its overall forecasts are not.

### 3.5 Final decision: XGBoost, with the alert threshold in the service

**Chosen model: XGBoost.** It has the best ranking of risky days (macro AUC 0.942, the best in all five years) and it passes the bar (+0.021 over persistence, ahead in 5 of 5 years). Its macro F1 is slightly below the forest's (0.655 against 0.664). The practical reasons favour it over the forest:

- a 4.2 MB model file against 114 MB;
- slightly faster predictions;
- half the Severe false alarms in 2023.

**The threshold is not tuned inside the model** (instead of the threshold step in the recommendation above):

- The model's category is the most likely one, the rule that was validated.
- The service also returns **P(Severe)** and **P(Dangerous)** = P(Very Poor) + P(Severe).
- The **alert threshold is a setting of the service**. It can be set, and changed later, once the users agree on the balance between Severe days caught and false alarms (Decision 4 in `model_decisions.md`), without retraining.
- Any threshold must be chosen on 2019–2023, **never on 2025**, so the 2025 test stays an honest check.

**Final training and test** (`scripts/train.py`):

- **Train:** 2015–2024, 43,755 city-days, with exactly the settings in section 4.4.
- **Test:** 2025, 4,877 city-days, scored **once**.
- The test is smaller than the planned 2024–2025, so Severe results on it will be noisier.

---

## 4. Tuning parameters of each model

**General rules used for every model:**

- **Settings were tuned on 2023.** When the top settings were too close to separate (within about 0.01), a shortlist was re-scored over five years, and the five-year result chose.
- **Tie-break:** a gap of 0.002 or less is a tie. Ties go to the setting that is ahead of persistence in more years, then to the simpler model (smaller training gap or fewer leaves).
- **Class weights were tried for every model,** because their effect differs from model to model.

### 4.1 Logistic regression

| Parameter      | What it controls                                                | Values tried                         | **Chosen**                                                                                  | How it was chosen                                                               |
| -------------- | --------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `class_weight` | Make rare categories count more                                 | `'balanced'`, `None`                 | **`None`**                                                                                  | 2023 macro F1 0.647 against 0.598; balanced said "Severe" 122 times for 51 real |
| `C`            | Inverse of L2 regularisation strength (small C = simpler model) | 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 | **1** (the default)                                                                         | Score rises until C = 1, then is flat                                           |
| `penalty`      | Type of regularisation                                          | L2 (default)                         | **L2**                                                                                      | Fixed                                                                           |
| `max_iter`     | Steps allowed for the solver to converge                        | —                                    | **3000**                                                                                    | Fixed, large enough to converge                                                 |
| Scaling        | Put all features on the same scale                              | `StandardScaler`                     | **Yes**                                                                                     | Required: AQI (0–500) vs 0/1 flags                                              |
| Features       | What the model sees                                             | 15 numeric + 7 yes/no + `city`       | `pressure_rel`, `precip_log`, `month_sin`/`month_cos`; `aqi_roll7_mean` dropped (VIF ≈ 341) | Transforms for a linear model                                                   |

2023 macro F1 for each C:

| C                | 0.0001 | 0.001 | 0.01  | 0.1   | **1**     | 10    | 100   |
| ---------------- | ------ | ----- | ----- | ----- | --------- | ----- | ----- |
| No class weights | 0.389  | 0.478 | 0.567 | 0.627 | **0.647** | 0.645 | 0.643 |
| Balanced         | 0.364  | 0.446 | 0.542 | 0.587 | 0.598     | 0.601 | 0.601 |

### 4.2 Decision tree

| Parameter          | What it controls                           | Values tried                       | **Chosen**                             | How it was chosen                                                               |
| ------------------ | ------------------------------------------ | ---------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------- |
| `max_depth`        | How many questions in a row                | 3, 5, 8, 10, 12, unlimited         | **10**                                 | Best 2023 score; deeper gives the same score                                    |
| `min_samples_leaf` | Minimum training days in every final group | 1, 50, 100, 200, 400               | **100**                                | Stops memorising: the unlimited tree scored 1.000 on training and 0.513 on 2023 |
| `class_weight`     | Make rare categories count more            | `None`, `'balanced'`               | **`None`**                             | Best balanced tree 0.636 against 0.658                                          |
| `criterion`        | How a split is judged                      | Gini (default)                     | **Gini**                               | Fixed                                                                           |
| Features           | What the model sees                        | 19 raw numeric + 7 yes/no + `city` | All four trend features, no transforms | Trees do not need scaling or transforms                                         |

60 settings in total; the chosen tree has 204 leaves. 2023 macro F1 without class weights:

| `max_depth` \ `min_samples_leaf` | 1     | 50    | **100**   | 200   | 400   |
| -------------------------------- | ----- | ----- | --------- | ----- | ----- |
| 3                                | 0.562 | 0.562 | 0.562     | 0.562 | 0.562 |
| 5                                | 0.643 | 0.643 | 0.643     | 0.643 | 0.651 |
| 8                                | 0.633 | 0.638 | 0.654     | 0.650 | 0.648 |
| **10**                           | 0.619 | 0.639 | **0.658** | 0.650 | 0.648 |
| 12                               | 0.593 | 0.638 | 0.658     | 0.650 | 0.648 |
| unlimited                        | 0.512 | 0.624 | 0.658     | 0.650 | 0.648 |

Five-year check of the neighbouring settings (leaf 50, 200, 400; depth 5): mean 0.623–0.631, none better than the chosen tree.

### 4.3 Random forest

| Parameter          | What it controls                                          | Values tried                         | **Chosen**                 | How it was chosen                                                            |
| ------------------ | --------------------------------------------------------- | ------------------------------------ | -------------------------- | ---------------------------------------------------------------------------- |
| `n_estimators`     | Number of trees that vote                                 | 1, 5, 10, 25, 50, 100, 200, 300, 500 | **200**                    | 2023 score flat from about 100 trees on (0.670–0.677); more trees never hurt |
| `max_features`     | Share of features a tree may choose from at each question | `'sqrt'` (≈6 of 40), 0.33, 0.5, 1.0  | **0.5**                    | Five-year shortlist                                                          |
| `min_samples_leaf` | Minimum training days in every leaf                       | 1, 5, 20, 50                         | **5**                      | Five-year shortlist; 50 is too coarse                                        |
| `class_weight`     | Make rare categories count more, per tree sample          | `None`, `'balanced_subsample'`       | **`'balanced_subsample'`** | Tied on 2023 (0.679 each), but +0.012 over five years                        |
| `max_depth`        | Depth limit                                               | unlimited                            | **unlimited**              | Fixed; the leaf size and the vote control overfitting                        |
| `bootstrap`        | Each tree sees a random sample of days                    | `True` (default)                     | **True**                   | Fixed                                                                        |
| `random_state`     | Same random samples on every run                          | 1                                    | **1**                      | Fixed                                                                        |

32 settings were tried on 2023 (4 × 4 × 2). The 10 best were within 0.005 (0.674–0.679), so the 3 best of each weighting went to the five-year check:

| Setting                                  | 2023 macro F1 | Five-year macro F1 | Years ahead | Training gap |
| ---------------------------------------- | ------------- | ------------------ | ----------- | ------------ |
| **balanced, `max_features` 0.5, leaf 5** | 0.679         | **0.664**          | **5 of 5**  | 0.230        |
| balanced, `max_features` 1.0, leaf 5     | 0.676         | 0.655              | 5 of 5      | 0.246        |
| no weights, `max_features` 0.33, leaf 20 | 0.678         | 0.652              | 4 of 5      | 0.074        |
| no weights, `max_features` 0.33, leaf 5  | 0.679         | 0.651              | 4 of 5      | 0.215        |
| no weights, `max_features` sqrt, leaf 5  | 0.677         | 0.647              | 4 of 5      | 0.200        |
| balanced, `max_features` sqrt, leaf 1    | 0.674         | 0.644              | 4 of 5      | 0.356        |

### 4.4 XGBoost (gradient boosted trees)

| Parameter                                   | What it controls                                          | Values tried                                                                              | **Chosen** | How it was chosen                       |
| ------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------- | --------------------------------------- |
| `n_estimators`                              | Number of correction rounds; each round adds one tree per category (6) | Up to 1,500, with **early stopping** after 50 rounds without improvement in 2023 log loss | **628** (3,768 trees) | Early stopping                          |
| `learning_rate`                             | Share of each correction kept                             | 0.05                                                                                      | **0.05**   | Fixed (smaller = steadier, more rounds) |
| `max_depth`                                 | Questions per tree                                        | 3, 5, 7                                                                                   | **3**      | Five-year shortlist + tie-break         |
| `min_child_weight`                          | Minimum evidence in a leaf: the sum over its days of 2·p·(1−p), where p is the current probability for that tree's category. Days the model is already sure about count almost nothing | 1, 10                                                                                     | **10**     | Five-year shortlist; lower training gap at every depth |
| Class weights (`balanced` in `CategoryXGB`) | Weight each day by its category                           | No, yes                                                                                   | **No**     | 2023: 0.668 against at best 0.663       |
| `subsample`                                 | Share of days each tree sees                              | 0.8                                                                                       | **0.8**    | Fixed                                   |
| `colsample_bytree`                          | Share of features each tree sees                          | 0.8                                                                                       | **0.8**    | Fixed                                   |
| `reg_lambda`                                | L2 penalty on leaf values                                 | 1 (default)                                                                               | **1**      | Fixed                                   |
| `objective` / `eval_metric`                 | What is minimised / watched                               | `multi:softprob` / `mlogloss`                                                             | —          | Fixed (six-category probabilities)      |
| `tree_method`                               | Training algorithm                                        | `hist`                                                                                    | —          | Fixed (fast)                            |

Learning curve, depth 5: the 2023 score peaks around 400 rounds (0.674), then falls (0.659 at 1,000 rounds) while the training score keeps rising (0.922). Too many rounds overfit, which is why early stopping is needed.

12 settings were tried on 2023 (3 × 2 × 2). The 3 best of each weighting went to the five-year check:

| Setting                                        | Rounds (early stopping) | 2023 macro F1 | Five-year macro F1 | Years ahead | Training gap |
| ---------------------------------------------- | ---------------------- | ------------- | ------------------ | ----------- | ------------ |
| **no weights, depth 3, `min_child_weight` 10** | 628                    | 0.668         | 0.655              | **5 of 5**  | **0.086**    |
| no weights, depth 7, `min_child_weight` 10     | 228                    | 0.678         | **0.656**          | 4 of 5      | 0.152        |
| no weights, depth 5, `min_child_weight` 1      | 407                    | 0.674         | 0.651              | 4 of 5      | 0.185        |
| balanced, depth 7, `min_child_weight` 10       | 693                    | 0.663         | 0.654              | 5 of 5      | 0.281        |
| balanced, depth 5, `min_child_weight` 1        | 1,134                  | 0.653         | 0.647              | 5 of 5      | 0.304        |
| balanced, depth 7, `min_child_weight` 1        | 584                    | 0.655         | 0.646              | 4 of 5      | 0.330        |

The top two are a tie (0.0009 apart). The tie-break chose depth 3, which is ahead of persistence in all five years and has the smallest training gap.

### 4.5 Class weights: the same setting, different effects

| Model               | Better without weights or with? | On 2023                | Over five years        |
| ------------------- | ------------------------------- | ---------------------- | ---------------------- |
| Logistic regression | Without                         | 0.647 vs 0.598         | 0.613 vs 0.614 (a tie) |
| Decision tree       | Without                         | 0.658 vs 0.636         | —                      |
| Random forest       | **With** (`balanced_subsample`) | 0.679 vs 0.679 (a tie) | **0.664 vs 0.652**     |
| XGBoost             | Without                         | 0.668 vs 0.663         | 0.655 vs 0.654 (a tie) |

**Lesson:** a setting like class weights must be checked for each model, and on five years.

---

## 5. Strengths and weaknesses at a glance

| Model               | Strengths                                                                                                                       | Weaknesses                                                                                                 | Verdict                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Logistic regression | Most explainable, tiny, no overfitting, best change-day score (35%), very good Severe ranking (AUC 0.981)                       | Below persistence over five years (0.613); unstable between years (spread 0.051); needs feature transforms | Not a candidate; useful as an explainable reference |
| Decision tree       | Explainable as questions, found the CPCB limits (50/100/200/300) by itself, best Severe AUC on 2023                             | Ties persistence over five years (0.631); coarse probabilities (lowest AUC, 0.930)                         | Not a candidate                                     |
| Random forest       | Best macro F1 (0.664, all five years ahead, best in 2022); best Dangerous recall (84%); catches the most Severe days (31 of 51) | Most Severe false alarms (35, of which 30 were Very Poor days); 114 MB file                                | Finalist, not chosen (file size)                    |
| XGBoost             | Best ROC AUC in every year (0.942); fewer Severe false alarms (18); most precise Dangerous alert (77%); 4.2 MB                  | Macro F1 a little lower than the forest's (0.655); needs early stopping                                    | **Chosen**: best by AUC                             |

**Next steps:**

1. Run `train.py`: train XGBoost on 2015–2024 and test it **once** on 2025.
2. Write `predict.py`: return the category, P(Severe) and P(Dangerous), with the alert threshold as a setting.
3. Build the web service and the Docker image.
