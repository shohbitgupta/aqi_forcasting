"""Train the final next-day AQI category model (XGBoost) and test it once on the held-out year.

    python scripts/train.py                      # train on 2015-2024, test on 2025, save models/xgb_final.bin
    python scripts/train.py --output my_model.bin

The settings below were chosen in xgb_train.ipynb with rolling validation on 2019-2023
(see docs/model_comparison.md, section 4.4). They are fixed here: nothing in this script
is chosen by looking at the test year, so the test stays an honest check.

Metrics (the classes are imbalanced: Severe is 2.4% of days): macro F1 is the main score,
then Dangerous (Very Poor or Severe) recall and precision; PR AUC and ROC AUC judge how well
the probabilities rank risky days; log loss, the quantity the model is trained to minimise,
judges the probabilities themselves; accuracy is shown for comparison only.
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from utility import (AQI_CATEGORIES, prepare_dataset, CategoryXGB, forecast_scorecard,
                     class_scores_table, proba_table, persistence_proba, auc_scores,
                     pr_auc_scores, class_share, log_loss_score)

PROJECT_DIR = Path(__file__).resolve().parent.parent     # works from any folder, not only scripts/
DATA_PATH = PROJECT_DIR / 'data' / 'processed' / 'aqi_weather_2015_2025.csv'
MODEL_PATH = PROJECT_DIR / 'models' / 'xgb_final.bin'

TARGET = 'tomorrow_status'
TREND = ['aqi_roll7_mean', 'aqi_roll7_std', 'aqi_diff1', 'aqi_dev7']
TRAIN_END, TEST_YEAR = 2024, 2025

# The same raw features as every tree model (no transforms needed)
NUMERIC = (['aqi_today'] + TREND +
           ['temp_mean', 'temp_max', 'temp_min', 'temp_range', 'humidity_mean', 'dew_point_mean',
            'precip_sum', 'pressure_mean', 'cloud_cover_mean', 'wind_speed_mean', 'wind_speed_max',
            'wind_u_mean', 'wind_v_mean', 'month'])
BINARY = ['has_co', 'has_no2', 'has_o3', 'has_pm10', 'has_pm25', 'has_so2', 'aqi_imputed']
CATEGORICAL = ['city']
FEATURES = NUMERIC + BINARY + CATEGORICAL

# Chosen in xgb_train.ipynb: depth 3, min_child_weight 10, no class weights,
# 628 rounds from early stopping on 2023 (kept fixed: never early-stop on the test year)
PARAMS = {
    'n_estimators': 628,
    'learning_rate': 0.05,
    'max_depth': 3,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'balanced': False,
}

# Rolling validation 2019-2023, to compare the test result with
VALIDATION_REFERENCE = {'macro_f1': 0.655, 'persistence_macro_f1': 0.634, 'macro_auc': 0.942}


def load_data(path=DATA_PATH):
    df = prepare_dataset(path)
    missing = [c for c in FEATURES + [TARGET, 'status_today', 'date'] if c not in df.columns]
    if missing:
        raise KeyError(f'columns missing from the data: {missing}')
    return df[df[TREND].notna().all(axis=1)]             # the same rows dropped as for every model


def split_by_year(df, train_end=TRAIN_END, test_year=TEST_YEAR):
    year = df['date'].dt.year
    full_train, test = df[year <= train_end], df[year == test_year]
    if full_train.empty or test.empty:
        raise ValueError(f'empty split: train up to {train_end}, test {test_year}')
    assert full_train['date'].max() < test['date'].min(), 'test days must come after training days'
    return full_train, test


def to_records(part):
    return part[FEATURES].to_dict(orient='records')


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=AQI_CATEGORIES, average='macro')


def make_model(params=PARAMS):
    return Pipeline([
        ('dv', DictVectorizer(sparse=False)),            # city -> one-hot; numbers pass through
        ('xgb', CategoryXGB(**params)),                  # takes and returns the category names
    ])


def train(full_train):
    y_train = full_train[TARGET].astype(str).to_numpy()
    model = make_model().fit(to_records(full_train), y_train)
    return model, macro_f1(y_train, model.predict(to_records(full_train)))


def evaluate(model, test):
    """Score the test year once: the model next to persistence (tomorrow = today)."""
    X_test = to_records(test)
    y_test = test[TARGET].astype(str).to_numpy()
    today = test['status_today'].astype(str).to_numpy()

    predictions = {'persistence': today, 'xgboost': model.predict(X_test)}
    probas = {'persistence': persistence_proba(test['status_today']), 'xgboost': proba_table(model, X_test)}
    auc = pd.DataFrame({name: auc_scores(y_test, p) for name, p in probas.items()}).T
    pr_auc = pd.DataFrame({'guessing (share of days)': class_share(y_test),
                           **{name: pr_auc_scores(y_test, p) for name, p in probas.items()}}).T
    shares = class_share(y_test)[AQI_CATEGORIES]
    guessing = pd.DataFrame([shares.to_numpy()] * len(y_test), columns=AQI_CATEGORIES)
    # persistence is left out: it gives 0 to every other category, so each miss costs infinity
    log_loss = {'guessing (share of days)': log_loss_score(y_test, guessing),
                'xgboost': log_loss_score(y_test, probas['xgboost'])}
    return {
        'scorecard': forecast_scorecard(y_test, predictions, y_today=test['status_today']),
        'class_scores': class_scores_table(y_test, predictions).round(2),
        'confusion': pd.DataFrame(confusion_matrix(y_test, predictions['xgboost'], labels=AQI_CATEGORIES),
                                  index=pd.Index(AQI_CATEGORIES, name='actual'),
                                  columns=pd.Index(AQI_CATEGORIES, name='predicted')),
        'auc': auc.round(3),
        'pr_auc': pr_auc.round(3),
        'log_loss': {name: round(v, 3) for name, v in log_loss.items()},
    }


def save(model, results, full_train, path):
    bundle = {
        'model': model,                                  # DictVectorizer + CategoryXGB (utility.py)
        'params': PARAMS,
        'features': {'numeric': NUMERIC, 'binary': BINARY, 'categorical': CATEGORICAL},
        'categories': AQI_CATEGORIES,
        'cities': sorted(full_train['city'].unique()),   # the service must reject any other city
        'train_years': (int(full_train['date'].dt.year.min()), TRAIN_END),
        'test_year': TEST_YEAR,
        'test_scores': results['scorecard']['xgboost'].to_dict(),
        'test_auc': results['auc'].loc['xgboost'].to_dict(),
        'test_pr_auc': results['pr_auc'].loc['xgboost'].to_dict(),
        'test_log_loss': results['log_loss']['xgboost'],
        'validation_reference': VALIDATION_REFERENCE,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    return path


def check_saved(path, test):
    """Reload the file and forecast one day as a dictionary, the way the web service will."""
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    one_day = to_records(test.head(1))
    probs = proba_table(loaded['model'], one_day).iloc[0]
    print(f"\nReload check, {test['city'].iloc[0]} on {test['date'].iloc[0]:%Y-%m-%d}: "
          f"forecast {probs.idxmax()}, P(Dangerous) {probs['Very Poor'] + probs['Severe']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', default=MODEL_PATH, help=f'where to save the model (default: {MODEL_PATH})')
    args = parser.parse_args()

    df = load_data()
    full_train, test = split_by_year(df)
    print(f'Train: {full_train["date"].min():%Y-%m-%d} to {full_train["date"].max():%Y-%m-%d}, {len(full_train):,} rows')
    print(f'Test:  {test["date"].min():%Y-%m-%d} to {test["date"].max():%Y-%m-%d}, {len(test):,} rows')

    model, train_f1 = train(full_train)
    print(f'\nMacro F1 on the training years: {train_f1:.3f}')

    results = evaluate(model, test)
    print(f'\n=== Test year {TEST_YEAR} (scored once) ===')
    print(results['scorecard'].to_string())
    print('\nPrecision / recall / F1 per category:')
    print(results['class_scores'].to_string())
    print('\nConfusion matrix (XGBoost):')
    print(results['confusion'].to_string())
    print('\nPR AUC (average precision; guessing scores the share of days):')
    print(results['pr_auc'].to_string())
    print('\nROC AUC:')
    print(results['auc'].to_string())
    print('\nLog loss (what training minimises; lower is better; persistence: infinite):')
    for name, value in results['log_loss'].items():
        print(f'  {name:26} {value:.3f}')
    ref = VALIDATION_REFERENCE
    print(f"\nFor comparison, rolling validation 2019-2023: macro F1 {ref['macro_f1']:.3f} "
          f"(persistence {ref['persistence_macro_f1']:.3f}), macro AUC {ref['macro_auc']:.3f}")

    path = save(model, results, full_train, args.output)
    print(f'\nSaved: {path}')
    check_saved(path, test)


if __name__ == '__main__':
    main()
