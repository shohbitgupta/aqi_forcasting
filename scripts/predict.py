"""Predict the next day's AQI category for a city with the final model: a web service and a command line.

Web service (FastAPI): POST one city-day's 27 fields to /predict, get the forecast, probabilities and alert;
or POST only {"city": "Delhi"} to /predict?live=true, and the service builds the fields from the latest
CPCB bulletins itself.

    uvicorn predict:app --app-dir scripts --port 8080          # then open http://localhost:8080/docs

Command line:

    python scripts/predict.py --input day.json                                 # your own input: one record or a list
    python scripts/predict.py --mode offline --city Delhi --date 2025-11-15    # a day from the project's data
    python scripts/predict.py --mode live --city Delhi                         # yesterday's CPCB bulletins

offline: the input comes from data/processed (2015-2025); --date defaults to the city's last day.
live:    the input is built from the CPCB bulletins (see validate_live.py); --date defaults to yesterday.
In both modes --city is asked for when left out, and for a past day the forecast is shown
next to what actually happened.
"""
import argparse
import json
import pickle
import sys
from contextlib import asynccontextmanager
from typing import Annotated
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Discriminator, Field, Tag

import validate_live as live
from utility import aqi_to_category, prepare_dataset, proba_table   # the pickle also needs utility (CategoryXGB)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / 'models' / 'xgb_final.bin'
DATA_PATH = PROJECT_DIR / 'data' / 'processed' / 'aqi_weather_2015_2025.csv'
OUTPUT_DIR = PROJECT_DIR / 'outputs' / 'validation'        # the inputs built in live mode
DANGEROUS = ['Very Poor', 'Severe']


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def feature_names(bundle):
    return bundle['features']['numeric'] + bundle['features']['binary'] + bundle['features']['categorical']


def find_city(name, bundle):
    """The model's spelling of a city name typed in any case, or None if the model does not know it."""
    return {c.lower(): c for c in bundle['cities']}.get(name.strip().lower())


def check_record(record, bundle):
    """Stop with a clear message if a record cannot be scored."""
    missing = [f for f in feature_names(bundle) if f not in record]
    if missing:
        raise ValueError(f'missing fields: {missing}')
    if record['city'] not in bundle['cities']:
        raise ValueError(f"unknown city {record['city']!r}; known cities: {bundle['cities']}")


def predict(records, bundle):
    """One result per record: forecast, probabilities, P(Severe), P(Dangerous)."""
    probs = proba_table(bundle['model'], records)           # one row per record, columns Good ... Severe
    results = []
    for record, (_, p) in zip(records, probs.iterrows()):
        results.append({
            'city': record['city'],
            'forecast': p.idxmax(),
            'p_severe': round(float(p['Severe']), 4),
            'p_dangerous': round(float(p['Very Poor'] + p['Severe']), 4),
            'probabilities': {c: round(float(p[c]), 4) for c in bundle['categories']},
        })
    return results

def decide_alert(result, severe_threshold=None, dangerous_threshold=None):
    if severe_threshold is None and dangerous_threshold is None:
        if result['forecast'] in DANGEROUS:
            return True, 'forecast is Very Poor or Severe'
        return False, 'forecast is below Very Poor'
    if severe_threshold is not None and result['p_severe'] >= severe_threshold:
        return True, f'P(Severe) >= {severe_threshold}'
    if dangerous_threshold is not None and result['p_dangerous'] >= dangerous_threshold:
        return True, f'P(Dangerous) >= {dangerous_threshold}'
    return False, 'below the alert thresholds'


# Web service ---------------------------------------------------------------------------------------

EXAMPLE = {        # demo.json: Noida on 2026-10-28, shown pre-filled on the /docs page
    'city': 'Noida', 'month': 10, 'aqi_today': 246.0, 'aqi_roll7_mean': 190.86, 'aqi_roll7_std': 43.41,
    'aqi_diff1': 82.0, 'aqi_dev7': 54.71, 'temp_mean': 23.3, 'temp_max': 29.3, 'temp_min': 17.0,
    'temp_range': 12.3, 'humidity_mean': 65.67, 'dew_point_mean': 15.89, 'precip_sum': 0.0,
    'pressure_mean': 992.0, 'cloud_cover_mean': 30.75, 'wind_speed_mean': 0.96, 'wind_speed_max': 1.51,
    'wind_u_mean': 0.1, 'wind_v_mean': -0.89, 'has_co': 0, 'has_no2': 0, 'has_o3': 0, 'has_pm10': 0,
    'has_pm25': 1, 'has_so2': 0, 'aqi_imputed': 0,
}
Flag = Field(ge=0, le=1)


class CityDay(BaseModel):
    """One city-day: the model's 27 input fields. FastAPI rejects a missing field or a wrong type."""
    model_config = {'extra': 'forbid', 'json_schema_extra': {'examples': [EXAMPLE]}}

    city: str
    month: int = Field(ge=1, le=12)
    aqi_today: float = Field(ge=0, description="the day's AQI, from the CPCB bulletin")
    aqi_roll7_mean: float = Field(ge=0, description='mean AQI of the last 7 days, the day included')
    aqi_roll7_std: float = Field(ge=0, description='standard deviation over the same 7 days')
    aqi_diff1: float = Field(description='the AQI minus the day before')
    aqi_dev7: float = Field(description='the AQI minus the mean of the 7 days before it')
    temp_mean: float
    temp_max: float
    temp_min: float
    temp_range: float = Field(ge=0)
    humidity_mean: float = Field(ge=0, le=100)
    dew_point_mean: float
    precip_sum: float = Field(ge=0)
    pressure_mean: float = Field(gt=0)
    cloud_cover_mean: float = Field(ge=0, le=100)
    wind_speed_mean: float = Field(ge=0)
    wind_speed_max: float = Field(ge=0)
    wind_u_mean: float
    wind_v_mean: float
    has_co: int = Flag
    has_no2: int = Flag
    has_o3: int = Flag
    has_pm10: int = Flag
    has_pm25: int = Flag
    has_so2: int = Flag
    aqi_imputed: int = Field(ge=0, le=1, description='1 if the AQI was filled from an earlier day')


class LiveCity(BaseModel):
    """With live=true, only the city: the service builds the other fields from the latest real data."""
    model_config = {'extra': 'forbid'}

    city: str = Field(description='one of the 14 cities, in any case, e.g. Delhi')


def body_kind(body):
    """'live' for a body with only the city, 'fields' otherwise: a wrong body is then checked
    against one shape only, so the error lists what is wrong with it and nothing else."""
    keys = set(body) if isinstance(body, dict) else set(type(body).model_fields)
    return 'live' if keys == {'city'} else 'fields'


Body27OrCity = Annotated[Annotated[CityDay, Tag('fields')] | Annotated[LiveCity, Tag('live')],
                         Discriminator(body_kind), Body(openapi_examples={
                             'fields': {'summary': 'live=false: the 27 fields (demo.json)', 'value': EXAMPLE},
                             'live': {'summary': 'live=true: only the city', 'value': {'city': 'Delhi'}}})]


@asynccontextmanager
async def lifespan(app):
    app.state.bundle = load_model()            # once, when the server starts: not on every request
    yield


app = FastAPI(title='Next-day AQI forecast', lifespan=lifespan,
              description="Tomorrow's CPCB air-quality category for 14 Indian cities.")


@app.get('/', include_in_schema=False)
def home():
    return RedirectResponse('/docs')                 # the interactive page, so a browser lands somewhere useful


@app.get('/health')
def health():
    return {'status': 'ok', 'cities': app.state.bundle['cities']}


@app.post('/predict')
def forecast(day: Body27OrCity,
             use_live: bool = Query(False, alias='live', description=(
                 'true: send only the city, and the service builds the 27 fields from the latest '
                 'CPCB bulletins and weather; false: send the 27 fields')),
             severe_threshold: float | None = Query(None, ge=0, le=1),
             dangerous_threshold: float | None = Query(None, ge=0, le=1)):
    """The next day's category, every category's probability, and whether to raise an alert.

    live=false (the default): the body is one city-day's 27 fields, as in demo.json.
    live=true: the body is only the city, and the forecast is for today. The service builds the
    27 fields the way `predict.py --mode live` does, from yesterday's CPCB bulletin, the 10 before
    it and yesterday's weather, and adds them to the answer with the dates. Bulletins are kept, so
    after the first live request only the newest one is downloaded; that first request takes longer.

    Without thresholds the alert is raised when the forecast is Very Poor or Severe. With them,
    it is raised when P(Severe) or P(Very Poor or Severe) reaches the threshold.
    """
    bundle = app.state.bundle
    if use_live != isinstance(day, LiveCity):
        raise HTTPException(status_code=422, detail=(
            'with live=true send only the city, e.g. {"city": "Delhi"}' if use_live else
            'only the city was sent: add ?live=true for a forecast from live data, or send all 27 fields'))
    if use_live:
        city = find_city(day.city, bundle)
        if city is None:
            raise HTTPException(status_code=422, detail=f"unknown city {day.city!r}; known cities: {bundle['cities']}")
        based_on = live.yesterday()                     # the last complete day: its bulletin is out
        try:
            record = live.build_record(city, based_on)
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f'could not reach CPCB: {e}')
        except RuntimeError as e:                       # too many missing bulletins, or no weather
            raise HTTPException(status_code=503, detail=str(e))
        live.prune_bulletins(live.window_start(based_on))
    else:
        record = day.model_dump()
        try:
            check_record(record, bundle)                # e.g. a city the model does not know
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    result = predict([record], bundle)[0]
    alert, rule = decide_alert(result, severe_threshold, dangerous_threshold)
    answer = {**result, 'alert': alert, 'alert_rule': rule}
    if use_live:
        answer.update(based_on=based_on, forecast_for=based_on + timedelta(days=1), input=record)
    return answer


# Command line --------------------------------------------------------------------------------------

def category(aqi):
    return str(aqi_to_category(pd.Series([aqi]))[0])


def pick_city(cities):
    """Ask for a city by number or name until the answer is one of the model's cities."""
    print('Supported cities:')
    for i, city in enumerate(cities, 1):
        print(f'  {i:>2}. {city}')
    by_name = {c.lower(): c for c in cities}
    while True:
        try:
            answer = input('Pick a city (number or name): ').strip()
        except EOFError:
            sys.exit('\nno city given: pass --city')
        if answer.isdigit() and 1 <= int(answer) <= len(cities):
            return cities[int(answer) - 1]
        if answer.lower() in by_name:
            return by_name[answer.lower()]
        print(f'  {answer!r} is not in the list')


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
    return day, record, (None, row['tomorrow_status'])


def live_case(city, day):
    """The input for `city` on `day` from the CPCB bulletins, and the AQI that followed if it is known."""
    day = day or live.yesterday()
    if day > live.yesterday():
        raise ValueError(f'--date must be a complete day: {live.yesterday()} or earlier')
    next_day = day + timedelta(days=1)

    print(f'Reading the CPCB bulletins for {city}, {day - timedelta(days=live.HISTORY_DAYS - 1)} to {day}...')
    record = live.build_record(city, day)
    actual = live.actual_aqi(city, next_day) if next_day <= live.yesterday() else None
    live.prune_bulletins(live.window_start(day))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{city.lower().replace(' ', '_')}_{day}.json"
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Saved the input to {path.relative_to(PROJECT_DIR)}\n')
    return day, record, None if actual is None else (actual, category(actual))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', help='JSON file: one city-day record or a list of them')
    source.add_argument('--mode', choices=['offline', 'live'],
                        help="offline: a day from the project's data; live: the CPCB bulletins")
    parser.add_argument('--city', help='with --mode: one of the supported cities; asked for when left out')
    parser.add_argument('--date', type=date.fromisoformat,
                        help='with --mode: the last day of data, YYYY-MM-DD')
    parser.add_argument('--model', default=MODEL_PATH)
    args = parser.parse_args()
    if args.input and (args.city or args.date):
        parser.error('--city and --date go with --mode, not --input')

    bundle = load_model(args.model)

    if args.input:
        records = json.loads(Path(args.input).read_text())
        if isinstance(records, dict):                # a single record is fine too
            records = [records]
        for record in records:
            check_record(record, bundle)
        for result in predict(records, bundle):
            print(f"{result['city']}: tomorrow's AQI is {result['forecast']}")
        return

    if args.city is None:
        city = pick_city(bundle['cities'])
    else:
        city = find_city(args.city, bundle)
        if city is None:
            parser.error(f"unknown city {args.city!r}; supported: {', '.join(bundle['cities'])}")

    try:
        case = offline_case(bundle, city, args.date) if args.mode == 'offline' else live_case(city, args.date)
    except (ValueError, RuntimeError) as e:
        sys.exit(f'error: {e}')
    day, record, actual = case
    check_record(record, bundle)
    result = predict([record], bundle)[0]

    next_day = day + timedelta(days=1)
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
