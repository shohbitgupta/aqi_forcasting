"""Data preparation, evaluation and model helpers shared by the scripts, the web service and the notebooks."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight

AQI_CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
AQI_BINS = [0, 50, 100, 200, 300, 400, float("inf")]


def aqi_to_category(aqi):
    """Map AQI values to CPCB categories (0-50 Good, 51-100 Satisfactory, ...)."""
    return pd.cut(aqi, bins=AQI_BINS, labels=AQI_CATEGORIES, include_lowest=True)


def ffill_aqi(df, col="aqi_today", limit=3):
    """Fill short AQI gaps with the last known value from the same city.

    Only past values are used, so nothing leaks from tomorrow. Gaps longer
    than `limit` days stay missing. Adds `aqi_imputed` (1 = filled) and
    recomputes `status_today` from the filled AQI so the two always agree.

    Create the next-day target BEFORE calling this, from the original values.
    """
    df = df.sort_values(["city", "date"]).copy()
    df["aqi_imputed"] = df[col].isna().astype(int)
    df[col] = df.groupby("city")[col].ffill(limit=limit)
    df.loc[df[col].isna(), "aqi_imputed"] = 0   # still missing, not filled
    df["status_today"] = aqi_to_category(df[col])
    return df


def split_train_val_test(df, date_col="date", train_end_year=2022, val_year=2023):
    """Split the data by time, so the model is always tested on later days.

    train : years up to and including train_end_year (default 2015-2022)
    val   : val_year                                 (default 2023)
    test  : every year after val_year                (default 2024-2025)

    A random split would put neighbouring days into both train and test,
    which makes next-day forecasting scores look better than they are.

    Returns (df_train, df_val, df_test), each sorted by city and date with a
    fresh index.
    """
    year = pd.to_datetime(df[date_col]).dt.year

    parts = (
        df[year <= train_end_year],
        df[year == val_year],
        df[year > val_year],
    )
    sort_cols = ["city", date_col] if "city" in df.columns else [date_col]
    return tuple(p.sort_values(sort_cols).reset_index(drop=True) for p in parts)


def class_box_stats(df, features, target="tomorrow_status", classes=AQI_CATEGORIES):
    """Q1, median and Q3 of each feature for each target class: the numbers
    behind a box plot. Returns a DataFrame indexed by (feature, stat)."""
    g = df.groupby(target, observed=True)[features]
    stats = {"q1": g.quantile(0.25), "median": g.median(), "q3": g.quantile(0.75)}
    out = pd.concat({k: v.reindex(classes).T for k, v in stats.items()}, names=["stat"])
    return out.swaplevel().sort_index(level=0, sort_remaining=False).round(2)


def box_overlap(a_q1, a_q3, b_q1, b_q3):
    """How much of the narrower box lies inside the other box.

    0 = the boxes are apart, 1 = the narrower box is completely inside the
    wider one. A zero-width box (e.g. rain = 0 on most days) counts as 1 if
    it sits inside the other box.
    """
    inter = max(0.0, min(a_q3, b_q3) - max(a_q1, b_q1))
    narrow = min(a_q3 - a_q1, b_q3 - b_q1)
    if narrow > 0:
        return inter / narrow
    point = a_q1 if a_q3 - a_q1 == 0 else b_q1        # the zero-width box
    other = (b_q1, b_q3) if a_q3 - a_q1 == 0 else (a_q1, a_q3)
    return float(other[0] <= point <= other[1])


def class_signal_summary(df, features, target="tomorrow_status", classes=AQI_CATEGORIES,
                         flat_tol=0.05, same_overlap=0.5, straight_r2=0.95):
    """Answer four questions about each feature's box plot across ordered classes.

    1. direction : do the medians move in a clear direction?
                   Spearman rank correlation (rho) between the feature and the
                   class order: |rho| < 0.1 'none', < 0.3 'weak up/down',
                   otherwise 'up' / 'down'.
    2. overlap   : do the boxes overlap? Average overlap of neighbouring boxes,
                   plus Good vs Severe (0 = apart, 1 = one box inside the other).
    3. shape     : is the pattern straight or does it bend?
                   'reverses' - the medians go one way, then back (e.g. a U-shape)
                   'straight' - a straight line fits the medians well
                                (R^2 >= straight_r2)
                   'bends'    - same direction throughout, but some steps are
                                flat or much steeper than others
    4. same_as   : which classes look the same? Neighbouring classes whose
                   median step is below `flat_tol` of the median range AND whose
                   boxes overlap more than `same_overlap`.

    Returns one row per feature, strongest signal (|rho|) first.
    """
    stats = class_box_stats(df, features, target, classes)
    order = {c: i for i, c in enumerate(classes)}
    codes = df[target].astype(str).map(order)
    idx = pd.Series(range(len(classes)), dtype=float)

    rows = []
    for f in features:
        med, q1, q3 = (stats.loc[(f, s)].to_numpy() for s in ("median", "q1", "q3"))
        steps = med[1:] - med[:-1]
        span = med.max() - med.min()
        flat = abs(steps) <= flat_tol * span
        moving = steps[~flat]

        r2 = pd.Series(med).corr(idx) ** 2 if span > 0 else 0.0
        if len(moving) == 0:
            shape = "flat"
        elif (moving > 0).any() and (moving < 0).any():
            shape = "reverses"
        elif r2 >= straight_r2:
            shape = "straight"
        else:
            shape = "bends"

        overlaps = [box_overlap(q1[i], q3[i], q1[i + 1], q3[i + 1])
                    for i in range(len(classes) - 1)]
        same = [f"{classes[i]}≈{classes[i + 1]}" for i in range(len(classes) - 1)
                if flat[i] and overlaps[i] > same_overlap]

        rho = df[f].rank().corr(codes.rank())   # Spearman = correlation of ranks
        strength = "" if abs(rho) >= 0.3 else "weak "
        rows.append({
            "feature": f,
            "rho": round(rho, 2),
            "direction": "none" if abs(rho) < 0.1 else strength + ("up" if rho > 0 else "down"),
            "medians": " → ".join(f"{m:g}" for m in med.round(1)),
            "avg_box_overlap": round(sum(overlaps) / len(overlaps), 2),
            "good_vs_severe_overlap": round(box_overlap(q1[0], q3[0], q1[-1], q3[-1]), 2),
            "linear_r2": round(r2, 2),
            "shape": shape,
            "same_as": ", ".join(same) or "-",
        })

    out = pd.DataFrame(rows).set_index("feature")
    return out.reindex(out["rho"].abs().sort_values(ascending=False).index)


def add_trend_features(df, col="aqi_today", window=7, min_periods=4, date_col="date"):
    """Add per-city lag and rolling AQI features, plus `month`.

    aqi_roll{w}_mean : mean AQI over the last w days, today included (t-w+1 .. t)
    aqi_roll{w}_std  : standard deviation over the same days (how unsettled the week was)
    aqi_diff1        : today minus yesterday (is AQI rising or falling?)
    aqi_dev{w}       : today minus the mean of the previous w days (t-w .. t-1),
                       i.e. how unusual today is. It leaves today out of the mean,
                       so it is not an exact combination of aqi_today and
                       aqi_roll{w}_mean.
    month            : calendar month (1-12)

    Every feature uses today and earlier days only, so nothing leaks from the
    target (tomorrow). Each city is placed on a full daily calendar first, so
    "yesterday" is always the previous calendar day even if rows were dropped:
    a missing day gives NaN, never a value from further back. A rolling window
    needs at least `min_periods` known days, otherwise it is NaN (the first
    days of each city, and around long gaps).

    Call it after `ffill_aqi` and BEFORE dropping rows: a dropped row's AQI is
    still a valid past value for the days that follow it.

    Returns a copy of df with the new columns; the index and row order are kept.
    """
    parts = []
    for city, s in df.set_index(date_col).groupby("city")[col]:
        s = s.sort_index().asfreq("D")              # one row per calendar day
        roll = s.rolling(window, min_periods=min_periods)
        prev_mean = s.shift(1).rolling(window, min_periods=min_periods).mean()
        parts.append(pd.DataFrame({
            "city": city,
            f"aqi_roll{window}_mean": roll.mean(),
            f"aqi_roll{window}_std": roll.std(),
            "aqi_diff1": s - s.shift(1),
            f"aqi_dev{window}": s - prev_mean,
        }))
    feats = pd.concat(parts).rename_axis(date_col).reset_index().set_index(["city", date_col])

    out = df.join(feats, on=["city", date_col])
    out["month"] = pd.to_datetime(out[date_col]).dt.month
    return out


def prepare_dataset(path, fill_limit=3):
    """Load the merged AQI + weather file and apply the EDA decisions, in the order that avoids leakage.

    1. Rename to aqi_today / status_today; rebuild the split later from `date`.
    2. Pollutant flags has_* (one 0/1 column per pollutant, NH3 dropped).
    3. Target tomorrow_status from the next day's AQI number, per city, BEFORE any filling.
    4. Forward fill input gaps of up to `fill_limit` days (adds aqi_imputed).
    5. Lag / rolling features + month (add_trend_features), BEFORE dropping rows.
    6. Drop rows without an AQI today or a real tomorrow; drop the station count.

    Returns one row per city-day. The trend features can still be NaN (first days of a
    city, around long gaps).
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"aqi_value": "aqi_today", "air_quality_status": "status_today"})
    df = df.drop(columns=["split"], errors="ignore")
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    flags = df["prominent_pollutants"].str.get_dummies(sep=", ")
    flags.columns = "has_" + flags.columns.str.lower().str.replace(".", "", regex=False)
    df = pd.concat([df, flags], axis=1).drop(columns=["prominent_pollutants", "has_nh3"], errors="ignore")

    df["tomorrow_status"] = aqi_to_category(df.groupby("city")["aqi_today"].shift(-1))
    df = ffill_aqi(df, limit=fill_limit)
    df = add_trend_features(df)

    df = df.dropna(subset=["aqi_today", "tomorrow_status"])
    df = df.drop(columns=["number_of_monitoring_stations"], errors="ignore")
    return df.reset_index(drop=True)


def add_lr_features(df, city_pressure):
    """Transforms that logistic regression needs (tree models do not).

    pressure_rel : pressure minus the city's normal pressure. Raw pressure mostly
                   encodes altitude (city normals differ by up to ~97 hPa); this keeps the weather part.
                   `city_pressure` must be the city means from the TRAINING split only.
    precip_log   : log1p(precip_sum), which shrinks the long rain tail (skew ~7.8 -> ~1.6).
    month_sin,
    month_cos    : month on a circle, so December and January are neighbours.
                   month_cos is high in winter, month_sin separates spring from autumn.
    """
    out = df.copy()
    out["pressure_rel"] = out["pressure_mean"] - out["city"].map(city_pressure)
    out["precip_log"] = np.log1p(out["precip_sum"])
    angle = 2 * np.pi * out["month"] / 12
    out["month_sin"], out["month_cos"] = np.sin(angle), np.cos(angle)
    return out


def vif_table(X):
    """VIF for each column of X. An intercept is added, as VIF requires.
    Rows with a missing or infinite value are skipped (and counted)."""
    import warnings
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    complete = np.isfinite(X).all(axis=1)
    if not complete.all():
        print(f"VIF: skipped {(~complete).sum():,} rows with missing values.")
    Xc = add_constant(X[complete])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # perfectly collinear sets warn about a singular matrix
        vif = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    return pd.Series(vif, index=X.columns, name="VIF").sort_values(ascending=False).round(2)



VP, SEV = AQI_CATEGORIES.index('Very Poor'), AQI_CATEGORIES.index('Severe')

def evaluate(name, y_true, y_pred):
    """Score one model's predictions. Returns one row for the results table."""
    y_true = pd.Series(y_true).astype(str).to_numpy()
    y_pred = pd.Series(y_pred).astype(str).to_numpy()

    kw = dict(labels=AQI_CATEGORIES, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, **kw)
    precision = precision_score(y_true, y_pred, average=None, **kw)
    return {
        'model': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', **kw),
        'recall_very_poor': recall[VP],
        'recall_severe': recall[SEV],
        'precision_very_poor': precision[VP],
        'precision_severe': precision[SEV],
    }


DANGEROUS = ["Very Poor", "Severe"]     # the health alert: Very Poor or worse


def forecast_scorecard(y_true, predictions, y_today, decimals=3):
    """Compare several forecasts on the same days. One column per forecast.

    y_true      : what really happened tomorrow
    predictions : {name: predicted categories}
    y_today     : today's category, to split days into "changed" and "unchanged"

    Rows: macro F1, accuracy, Severe caught / missed / false alarms,
    "Dangerous" alert recall and precision, and change / unchanged days right.
    Counts stay whole numbers; scores are rounded to `decimals`.
    """
    y_true = pd.Series(y_true).astype(str).to_numpy()
    changed = pd.Series(y_today).astype(str).to_numpy() != y_true
    real_dangerous = np.isin(y_true, DANGEROUS)

    table = {}
    for name, pred in predictions.items():
        pred = pd.Series(pred).astype(str).to_numpy()
        severe_hit = ((pred == "Severe") & (y_true == "Severe")).sum()
        alert = np.isin(pred, DANGEROUS)
        score = lambda x: round(float(x), decimals)
        table[name] = {
            "macro F1": score(f1_score(y_true, pred, labels=AQI_CATEGORIES, average="macro", zero_division=0)),
            "accuracy": score((pred == y_true).mean()),
            "Severe caught": int(severe_hit),
            "Severe missed": int((y_true == "Severe").sum() - severe_hit),
            "Severe false alarms": int((pred == "Severe").sum() - severe_hit),
            "Dangerous recall": score((alert & real_dangerous).sum() / real_dangerous.sum()),
            "Dangerous precision": score((alert & real_dangerous).sum() / alert.sum() if alert.any() else 0.0),
            "change days right": score((pred[changed] == y_true[changed]).mean()),
            "unchanged days right": score((pred[~changed] == y_true[~changed]).mean()),
        }
    # object columns, built per forecast, so counts stay whole numbers next to the scores
    return pd.DataFrame({name: pd.Series(rows, dtype=object) for name, rows in table.items()})


def class_scores_table(y_true, predictions):
    """Precision, recall and F1 for every category, side by side for several forecasts."""
    y_true = pd.Series(y_true).astype(str).to_numpy()
    parts = {}
    for name, pred in predictions.items():
        pred = pd.Series(pred).astype(str).to_numpy()
        kw = dict(labels=AQI_CATEGORIES, average=None, zero_division=0)
        parts[name] = pd.DataFrame({
            "precision": precision_score(y_true, pred, **kw),
            "recall": recall_score(y_true, pred, **kw),
            "F1": f1_score(y_true, pred, **kw),
        }, index=AQI_CATEGORIES)
    return pd.concat(parts, axis=1)


def proba_table(model, X):
    """The model's probability for every category, as a table with columns in AQI_CATEGORIES order.

    scikit-learn orders predict_proba columns by model.classes_ (alphabetical for text
    labels), so the columns are named and reordered here to avoid mixing them up.
    """
    return pd.DataFrame(model.predict_proba(X), columns=list(model.classes_))[AQI_CATEGORIES]


def persistence_proba(status_today):
    """Persistence as 'probabilities': 1 for today's category, 0 for every other one.

    Persistence only says yes or no, so it gets a single point on the ROC curve; its AUC
    is (recall + share of other days correctly left out) / 2, not a full ranking score.
    """
    status = pd.Series(np.asarray(status_today, dtype=str))
    return pd.DataFrame({c: (status == c).astype(float).to_numpy() for c in AQI_CATEGORIES})


def auc_scores(y_true, proba):
    """ROC AUC for every category (one against the rest), their average, and the Dangerous alert.

    AUC: take one day that really was Severe and one that was not; AUC is the chance the
    model gave the Severe day the higher Severe probability. 0.5 = coin toss, 1.0 = perfect.
    It judges the ranking of the probabilities, before any threshold is chosen.
    """
    y_true = np.asarray(y_true, dtype=str)
    scores = {c: roc_auc_score(y_true == c, proba[c]) for c in AQI_CATEGORIES}
    scores["macro"] = float(np.mean([scores[c] for c in AQI_CATEGORIES]))
    scores["Dangerous"] = roc_auc_score(np.isin(y_true, DANGEROUS), proba[DANGEROUS].sum(axis=1))
    return pd.Series(scores)


def pr_auc_scores(y_true, proba):
    """PR AUC (average precision) for every category, their average, and the Dangerous alert.

    The area under the precision-recall curve: how precise the alerts stay as the threshold
    is lowered to catch more of the real days. Unlike ROC AUC it ignores the many correctly
    ignored ordinary days, so it is the stricter ranking score for rare classes. A model that
    guesses scores the class's share of days (about 0.06 for Dangerous in 2025), not 0.5.
    """
    y_true = np.asarray(y_true, dtype=str)
    scores = {c: average_precision_score(y_true == c, proba[c]) for c in AQI_CATEGORIES}
    scores["macro"] = float(np.mean([scores[c] for c in AQI_CATEGORIES]))
    scores["Dangerous"] = average_precision_score(np.isin(y_true, DANGEROUS), proba[DANGEROUS].sum(axis=1))
    return pd.Series(scores)


def class_share(y_true):
    """Each category's share of days, the macro average and the Dangerous share: the PR AUC of guessing."""
    y_true = np.asarray(y_true, dtype=str)
    scores = {c: float((y_true == c).mean()) for c in AQI_CATEGORIES}
    scores["macro"] = float(np.mean([scores[c] for c in AQI_CATEGORIES]))
    scores["Dangerous"] = float(np.isin(y_true, DANGEROUS).mean())
    return pd.Series(scores)


def log_loss_score(y_true, proba):
    """Multi-class log loss: the mean of -ln(probability given to the category that happened).

    The quantity XGBoost minimises while training (objective multi:softprob). `proba` is a
    table with one column per category (proba_table): the true category's column is looked
    up by name. scikit-learn's log_loss instead assumes the columns are in alphabetical
    order (Good, Moderate, Poor, ...), which silently gives a wrong value for AQI order.
    """
    codes = pd.Categorical(np.asarray(y_true, dtype=str), categories=AQI_CATEGORIES).codes
    p_true = proba[AQI_CATEGORIES].to_numpy()[np.arange(len(codes)), codes]
    return float(-np.log(np.clip(p_true, 1e-15, 1)).mean())


def rolling_validation(df, make_model, to_xy, val_years=range(2019, 2024), test_from=2024,
                       target="tomorrow_status", date_col="date"):
    """Score a model on several years, each time learning only from the years before it.

    Round for val_year 2021: learn from every year < 2021, score on 2021 alone. So every
    round imitates real use (the model always predicts days after the ones it learned
    from), and together the rounds show whether a result is stable or luck.

    make_model : function returning a fresh, unfitted model
    to_xy      : function (train_part, val_part) -> (X_train, X_val). Anything learned from
                 data (e.g. city pressure normals) must come from train_part only.
    test_from  : the first test year; scoring it (or later) is refused, so the test set stays locked.

    Returns one row per year: the model's and persistence's macro F1 on that year, and
    the training-years macro F1 (a large gap means overfitting).
    """
    if any(year >= test_from for year in val_years):
        raise ValueError(f"val_years must be before {test_from}: the test years stay locked")

    year = pd.to_datetime(df[date_col]).dt.year
    rows = []
    for val_year in val_years:
        train, val = df[year < val_year], df[year == val_year]
        X_train, X_val = to_xy(train, val)
        y_train = train[target].astype(str).to_numpy()
        y_val = val[target].astype(str).to_numpy()

        model = make_model().fit(X_train, y_train)
        f1 = lambda y, p: f1_score(y, p, labels=AQI_CATEGORIES, average="macro", zero_division=0)
        rows.append({
            "val_year": val_year,
            "train_rows": len(train),
            "val_rows": len(val),
            "severe_days": int((y_val == "Severe").sum()),
            "train_macro_f1": f1(y_train, model.predict(X_train)),
            "macro_f1": f1(y_val, model.predict(X_val)),
            "persistence_macro_f1": f1(y_val, val["status_today"].astype(str).to_numpy()),
        })
        if hasattr(model, "predict_proba"):              # AUC needs probabilities, not just a category
            rows[-1]["macro_auc"] = auc_scores(y_val, proba_table(model, X_val))["macro"]
            rows[-1]["persistence_macro_auc"] = auc_scores(y_val, persistence_proba(val["status_today"]))["macro"]
    out = pd.DataFrame(rows)
    out["vs_persistence"] = out["macro_f1"] - out["persistence_macro_f1"]
    return out


def rolling_summary(rolling):
    """One row per model from stacked rolling_validation results (needs a 'model' column).

    passes_bar is the rule every new model must meet: beat persistence on the
    five-year average AND in most of the years, not just in one lucky year.
    """
    summary = rolling.groupby("model", sort=False).agg(
        mean_macro_f1=("macro_f1", "mean"),
        spread=("macro_f1", "std"),                     # how much the score moves from year to year
        mean_vs_persistence=("vs_persistence", "mean"),
        years_won=("vs_persistence", lambda d: int((d > 0).sum())),
        years=("vs_persistence", "size"),
        mean_train_macro_f1=("train_macro_f1", "mean"),
    )
    if "macro_auc" in rolling:                           # only when the models gave probabilities
        summary["mean_macro_auc"] = rolling.groupby("model", sort=False)["macro_auc"].mean()
    summary["mean_train_gap"] = summary.pop("mean_train_macro_f1") - summary["mean_macro_f1"]
    summary["passes_bar"] = (summary["mean_vs_persistence"] > 0) & (summary["years_won"] > summary["years"] / 2)
    return summary


class CategoryXGB(BaseEstimator, ClassifierMixin):
    """XGBoost that takes and returns the AQI category names.

    XGBoost only accepts the classes as numbers 0-5. This wrapper turns the names
    into numbers in AQI_CATEGORIES order before training, and back into names when
    predicting, so it fits in a Pipeline and in rolling_validation like any other model.

    balanced=True weights every training day by its class (like class_weight='balanced').
    fit(..., eval_set=[(X_val, y_val)], early_stopping_rounds=50) stops adding trees
    once the validation loss stops improving; best_n_trees_ then holds the number used.
    """

    def __init__(self, n_estimators=300, learning_rate=0.1, max_depth=6, min_child_weight=1,
                 subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, balanced=False, random_state=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.balanced = balanced
        self.random_state = random_state

    @staticmethod
    def _codes(y):
        return pd.Categorical(y, categories=AQI_CATEGORIES).codes      # name -> 0..5, unknown -> -1

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, sample_weight=None):
        """sample_weight: optional weight per training day (e.g. more for recent years), times the class weight."""
        from xgboost import XGBClassifier                               # imported here so utility works without xgboost

        codes = self._codes(y)
        if (codes < 0).any():
            raise ValueError(f"unknown categories in y: {sorted(set(np.asarray(y)[codes < 0]))}")
        self.classes_ = np.array(AQI_CATEGORIES)
        self.booster_ = XGBClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth,
            min_child_weight=self.min_child_weight, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
            objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
            early_stopping_rounds=early_stopping_rounds, random_state=self.random_state, n_jobs=-1)
        weights = compute_sample_weight("balanced", codes) if self.balanced else None
        if sample_weight is not None:
            weights = np.asarray(sample_weight) * (1 if weights is None else weights)
        evals = None if eval_set is None else [(X_e, self._codes(y_e)) for X_e, y_e in eval_set]
        self.booster_.fit(X, codes, sample_weight=weights, eval_set=evals, verbose=False)
        self.best_n_trees_ = (self.booster_.best_iteration + 1) if early_stopping_rounds else self.n_estimators
        return self

    def predict_proba(self, X):
        return self.booster_.predict_proba(X)                           # columns in AQI_CATEGORIES order

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
