"""Helper functions shared by the notebooks."""
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import medcouple

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


def skew_level(skew):
    """Name a skewness value: |skew| < 0.5 symmetric, < 1 moderate, otherwise heavy."""
    a = abs(skew)
    return "symmetric" if a < 0.5 else "moderate" if a < 1 else "heavy"


def outlier_fences(s, k=1.5, skew_threshold=1.0, spike_share=0.5):
    """Lower and upper outlier fences for one numeric Series, chosen by its shape.

    The 1.5 * IQR rule (Tukey) assumes a roughly symmetric distribution. On a
    skewed one it flags ordinary values in the long tail and misses real
    outliers on the short side. So:

    - |skew| <= skew_threshold: Tukey fences, Q1 - k*IQR .. Q3 + k*IQR.
    - |skew| >  skew_threshold: adjusted boxplot (Hubert & Vandervieren, 2008).
      The fences move with the medcouple (MC), a robust skewness measure:
        MC >= 0: Q1 - k*e^(-4MC)*IQR .. Q3 + k*e^(3MC)*IQR
        MC <  0: Q1 - k*e^(-3MC)*IQR .. Q3 + k*e^(4MC)*IQR
      The long side gets a wider fence and the short side a tighter one. MC
      is clipped to [-0.6, 0.6], the range the rule was calibrated on. With
      MC = 0 the fences are exactly Tukey's, so a symmetric body whose skew
      comes only from a few extremes is still judged like Tukey would.

    A value that covers at least `spike_share` of the data (e.g. 0 mm of rain
    on most days) is a spike, not an outlier: fences are computed on the other
    values and the spike itself is never flagged.

    Returns a dict: method, skew, skew_level, mc (adjusted boxplot only), spike,
    lower, upper.
    """
    x = pd.Series(s).dropna()
    spike = None
    shares = x.value_counts(normalize=True)
    if len(shares) and shares.iloc[0] >= spike_share:
        spike = shares.index[0]
        x = x[x != spike]

    out = {"method": "too few values", "skew": np.nan, "skew_level": "-", "mc": np.nan,
           "spike": spike, "lower": np.nan, "upper": np.nan}
    if len(x) < 4:
        return out

    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    skew = x.skew()

    if abs(skew) <= skew_threshold:
        method, mc, lo_mult, hi_mult = "tukey", np.nan, 1.0, 1.0
    else:
        method = "adjusted boxplot"
        mc = float(np.clip(medcouple(x.to_numpy(dtype=float)), -0.6, 0.6))
        a, b = (-4, 3) if mc >= 0 else (-3, 4)
        lo_mult, hi_mult = np.exp(a * mc), np.exp(b * mc)

    out.update(method=method, skew=round(skew, 2), skew_level=skew_level(skew), mc=round(mc, 3),
               lower=q1 - k * lo_mult * iqr, upper=q3 + k * hi_mult * iqr)
    return out


def detect_outliers(df, features, by=None, k=1.5, skew_threshold=1.0, spike_share=0.5):
    """Flag outliers in each feature with fences that respect its skew (see
    `outlier_fences`).

    by : optional column (e.g. 'city'). Fences are then computed within each
         group, so a value is judged against its own group's normal. Use it
         when groups sit at different levels (pressure by city altitude):
         pooled, they look like a skewed or multi-peaked distribution.

    Returns (summary, mask):
      summary : one row per feature (per feature and group with `by`) with the
                method, skew, medcouple, fences, low/high counts, % flagged,
                and `n_tukey`, what the plain 1.5 * IQR rule on all values
                would have flagged, for comparison.
      mask    : boolean DataFrame like df[features], True = outlier.
    """
    groups = df.groupby(by, observed=True) if by else [(None, df)]
    mask = pd.DataFrame(False, index=df.index, columns=features)
    rows = []

    for g, part in groups:
        for f in features:
            v = part[f]
            fz = outlier_fences(v, k, skew_threshold, spike_share)
            low, high = v < fz["lower"], v > fz["upper"]
            mask.loc[part.index, f] = low | high

            q1, q3 = v.quantile([0.25, 0.75])
            tukey = ((v < q1 - 1.5 * (q3 - q1)) | (v > q3 + 1.5 * (q3 - q1))).sum()
            rows.append({"feature": f, **({by: g} if by else {}), **fz,
                         "lower": round(fz["lower"], 2), "upper": round(fz["upper"], 2),
                         "n_low": int(low.sum()), "n_high": int(high.sum()),
                         "pct": round((low | high).mean() * 100, 2), "n_tukey": int(tukey),
                         "min": v.min(), "max": v.max()})

    summary = pd.DataFrame(rows).set_index(["feature", by] if by else "feature")
    if not by:
        summary = summary.sort_values("pct", ascending=False)
    return summary, mask


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
