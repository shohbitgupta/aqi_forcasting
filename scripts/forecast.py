"""Forecast the next day's AQI category for a city from the command line, with the final model.

    python scripts/forecast.py --mode offline --city Delhi --date 2025-11-15   # a day from the project's data
    python scripts/forecast.py --mode live --city Delhi                        # the latest CPCB bulletins
    python scripts/forecast.py --input demo.json                               # your own input: one city-day

offline: the input comes from data/processed (2015-2025); --date defaults to the city's last day.
live:    the input is built from the CPCB bulletins (see live_data.py); --date defaults to the latest
         bulletin: today's once it is out (about 4 PM IST), so the forecast is for tomorrow, otherwise
         yesterday's. It also shows how the model did on that day. The input is saved in outputs/validation/.
For a past day the forecast is shown next to what actually happened.
"""
import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import live_data
from predict import category, check_forecast, check_record, feature_names, find_city, load_model, predict
from utility import prepare_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / 'data' / 'processed' / 'aqi_weather_2015_2025.csv'
OUTPUT_DIR = PROJECT_DIR / 'outputs' / 'validation'        # the inputs built in live mode


def offline_case(bundle, city, day):
    """The input for `city` on `day` from the project's data, and the category that followed."""
    rows = prepare_dataset(DATA_PATH)
    rows = rows[rows['city'] == city].set_index('date')
    day = day or rows.index.max().date()
    if pd.Timestamp(day) not in rows.index:
        raise ValueError(f'no data for {city} on {day}: the data runs from {rows.index.min():%Y-%m-%d} '
                         f'to {rows.index.max():%Y-%m-%d}, with some days missing')
    row = rows.loc[pd.Timestamp(day)]
    record = {f: row[f] for f in feature_names(bundle)}
    if any(pd.isna(v) for v in record.values()):
        raise ValueError(f'{city} on {day}: too few days of AQI before it to compute the trend features')
    return day, record, (None, row['tomorrow_status']), None


def live_case(bundle, city, day):
    """The input for `city` on `day` from the CPCB bulletins, the AQI that followed if it is known,
    and how the model did on `day` itself (see predict.check_forecast)."""
    latest = live_data.latest_day()
    day = day or latest
    if day > latest:
        raise ValueError(f'--date must be a day with a bulletin: {latest} or earlier')
    next_day = day + timedelta(days=1)

    print(f'Reading the CPCB bulletins for {city}, {day - timedelta(days=live_data.HISTORY_DAYS - 1)} to {day}...')
    record = live_data.build_record(city, day)
    actual = live_data.actual_aqi(city, next_day) if next_day <= latest else None
    try:
        check = check_forecast(city, day, bundle)
    except RuntimeError as e:                            # the forecast does not depend on it
        check = {'date': day, 'error': str(e)}
    live_data.prune_bulletins(live_data.window_start(day - timedelta(days=1)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{city.lower().replace(' ', '_')}_{day}.json"
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Saved the input to {path.relative_to(PROJECT_DIR)}\n')
    return day, record, None if actual is None else (actual, category(actual)), check


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', help='JSON file with one city-day: the 27 fields, as in demo.json')
    source.add_argument('--mode', choices=['offline', 'live'],
                        help="offline: a day from the project's data; live: the CPCB bulletins")
    parser.add_argument('--city', help='with --mode: one of the 14 cities')
    parser.add_argument('--date', type=date.fromisoformat, help='with --mode: the last day of data, YYYY-MM-DD')
    args = parser.parse_args()

    bundle = load_model()

    if args.input:
        if args.city or args.date:
            parser.error('--city and --date go with --mode, not --input')
        record = json.loads(Path(args.input).read_text())
        try:
            check_record(record, bundle)
        except ValueError as e:
            sys.exit(f'error: {e}')
        print(f"{record['city']}: tomorrow's AQI is {predict(record, bundle)['forecast']}")
        return

    if args.city is None:
        parser.error(f"--mode needs --city, one of: {', '.join(bundle['cities'])}")
    city = find_city(args.city, bundle)
    if city is None:
        parser.error(f"unknown city {args.city!r}; supported: {', '.join(bundle['cities'])}")

    try:
        case = offline_case(bundle, city, args.date) if args.mode == 'offline' else live_case(bundle, city, args.date)
    except (ValueError, RuntimeError) as e:
        sys.exit(f'error: {e}')
    day, record, actual, check = case
    result = predict(record, bundle)

    next_day = day + timedelta(days=1)
    if check is not None:
        print(f'Check for {day}: ' + (
            f"not available ({check['error']})" if 'error' in check else
            f"forecast {check['forecast']}; {city} is missing from that day's bulletin" if check['actual'] is None else
            f"forecast {check['forecast']} (from the day before's data), actual AQI {check['actual_aqi']:.0f} "
            f"({check['actual']}): {'correct' if check['correct'] else 'missed'}"))
    filled = ', filled from an earlier day' if record['aqi_imputed'] else ''
    print(f"{city}, based on {day}: AQI {record['aqi_today']:.0f} ({category(record['aqi_today'])}{filled})")
    print(f"Forecast for {next_day}: {result['forecast']}")
    if actual is not None:
        aqi, status = actual
        verdict = 'correct' if status == result['forecast'] else 'missed'
        print(f"Actual on {next_day}:   {status if aqi is None else f'AQI {aqi:.0f} ({status})'}, forecast {verdict}")
    first_train, last_train = bundle['train_years']
    if args.mode == 'offline' and first_train <= day.year <= last_train:
        print(f"Note: the model was trained on {first_train}-{last_train}, so it has seen this day. "
              f"For a fair test pick a day in {bundle['test_year']}.")


if __name__ == '__main__':
    main()
