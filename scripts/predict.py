"""The prediction web service (FastAPI): the next day's AQI category for a city, with the final model.

    uvicorn predict:app --app-dir scripts --port 8080          # then open http://localhost:8080/docs

POST /predict with one city-day's 27 fields (see demo.json) returns the forecast, the probability of
every category and the alert. POST /predict?live=true with only {"city": "Delhi"} does the same from
the latest CPCB bulletin and weather, which the service fetches itself: after about 4 PM IST, when
today's bulletin is out, the forecast is for tomorrow. It also checks the model on the latest day.

The command line (scripts/forecast.py) uses the functions here too.
"""
import pickle
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import live_data
from utility import aqi_to_category, proba_table      # the pickle also needs utility (CategoryXGB)

MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'xgb_final.bin'
DANGEROUS = ['Very Poor', 'Severe']


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def feature_names(bundle):
    return bundle['features']['numeric'] + bundle['features']['binary'] + bundle['features']['categorical']


def find_city(name, bundle):
    """The model's spelling of a city name typed in any case, or None if the model does not know it."""
    return {c.lower(): c for c in bundle['cities']}.get(str(name).strip().lower())


def check_record(record, bundle):
    """Stop with a clear message if a record cannot be scored."""
    missing = [f for f in feature_names(bundle) if f not in record]
    if missing:
        raise ValueError(f'missing fields: {missing}')
    if record['city'] not in bundle['cities']:
        raise ValueError(f"unknown city {record['city']!r}; known cities: {bundle['cities']}")


def predict(record, bundle):
    """The forecast for one city-day: the category, its probabilities, P(Severe) and P(Dangerous)."""
    p = proba_table(bundle['model'], [record]).iloc[0]       # one probability per category, Good ... Severe
    return {
        'city': record['city'],
        'forecast': p.idxmax(),
        'p_severe': round(float(p['Severe']), 4),
        'p_dangerous': round(float(p['Very Poor'] + p['Severe']), 4),
        'probabilities': {c: round(float(p[c]), 4) for c in bundle['categories']},
    }


def category(aqi):
    return str(aqi_to_category(pd.Series([aqi]))[0])


def check_forecast(city, day, bundle):
    """How the model did on `day`: its forecast from the day before's data, against the bulletin's AQI."""
    forecast = predict(live_data.build_record(city, day - timedelta(days=1)), bundle)['forecast']
    aqi = live_data.actual_aqi(city, day)
    actual = None if aqi is None else category(aqi)          # None: the city is missing from that bulletin
    return {'date': day, 'forecast': forecast, 'actual_aqi': aqi, 'actual': actual,
            'correct': None if actual is None else forecast == actual}


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


# The request ---------------------------------------------------------------------------------------

EXAMPLE = {        # demo.json: Noida on 2026-10-28, shown pre-filled on the /docs page
    'city': 'Noida', 'month': 10, 'aqi_today': 246.0, 'aqi_roll7_mean': 190.86, 'aqi_roll7_std': 43.41,
    'aqi_diff1': 82.0, 'aqi_dev7': 54.71, 'temp_mean': 23.3, 'temp_max': 29.3, 'temp_min': 17.0,
    'temp_range': 12.3, 'humidity_mean': 65.67, 'dew_point_mean': 15.89, 'precip_sum': 0.0,
    'pressure_mean': 992.0, 'cloud_cover_mean': 30.75, 'wind_speed_mean': 0.96, 'wind_speed_max': 1.51,
    'wind_u_mean': 0.1, 'wind_v_mean': -0.89, 'has_co': 0, 'has_no2': 0, 'has_o3': 0, 'has_pm10': 0,
    'has_pm25': 1, 'has_so2': 0, 'aqi_imputed': 0,
}
Flag = Field(None, ge=0, le=1)


class CityDay(BaseModel):
    """One city-day. With live=true send only the city; otherwise all 27 fields are needed.
    FastAPI rejects a wrong type, a value out of range or an unknown field."""
    model_config = {'extra': 'forbid'}

    city: str = Field(description='one of the 14 cities, in any case, e.g. Delhi')
    month: int | None = Field(None, ge=1, le=12)
    aqi_today: float | None = Field(None, ge=0, description="the day's AQI, from the CPCB bulletin")
    aqi_roll7_mean: float | None = Field(None, ge=0, description='mean AQI of the last 7 days, the day included')
    aqi_roll7_std: float | None = Field(None, ge=0, description='standard deviation over the same 7 days')
    aqi_diff1: float | None = Field(None, description='the AQI minus the day before')
    aqi_dev7: float | None = Field(None, description='the AQI minus the mean of the 7 days before it')
    temp_mean: float | None = None
    temp_max: float | None = None
    temp_min: float | None = None
    temp_range: float | None = Field(None, ge=0)
    humidity_mean: float | None = Field(None, ge=0, le=100)
    dew_point_mean: float | None = None
    precip_sum: float | None = Field(None, ge=0)
    pressure_mean: float | None = Field(None, gt=0)
    cloud_cover_mean: float | None = Field(None, ge=0, le=100)
    wind_speed_mean: float | None = Field(None, ge=0)
    wind_speed_max: float | None = Field(None, ge=0)
    wind_u_mean: float | None = None
    wind_v_mean: float | None = None
    has_co: int | None = Flag
    has_no2: int | None = Flag
    has_o3: int | None = Flag
    has_pm10: int | None = Flag
    has_pm25: int | None = Flag
    has_so2: int | None = Flag
    aqi_imputed: int | None = Field(None, ge=0, le=1, description='1 if the AQI was filled from an earlier day')


EXAMPLES = {
    'fields': {'summary': 'the 27 fields (demo.json)', 'value': EXAMPLE},
    'live': {'summary': 'live=true: only the city', 'value': {'city': 'Delhi'}},
}


# The service ---------------------------------------------------------------------------------------

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
def forecast(day: CityDay = Body(openapi_examples=EXAMPLES),
             live: bool = Query(False, description='true: send only the city, and the service builds the '
                                                   'other fields from the latest CPCB bulletins and weather'),
             severe_threshold: float | None = Query(None, ge=0, le=1),
             dangerous_threshold: float | None = Query(None, ge=0, le=1)):
    """The next day's category, every category's probability, and whether to raise an alert.

    live=false (the default): the body is one city-day's 27 fields, as in demo.json.
    live=true: the body is only the city. The service builds the 27 fields from the latest CPCB
    bulletin (today's once it is out, about 4 PM IST, otherwise yesterday's), the 10 before it and that
    day's weather, exactly as the training data was built, and adds them to the answer with the dates.
    `check` shows how the model did on that latest day: its forecast from the day before's data,
    against the bulletin's actual AQI. Bulletins are kept, so after the first live request only the
    newest one is downloaded; that first request takes longer.

    Without thresholds the alert is raised when the forecast is Very Poor or Severe. With them,
    it is raised when P(Severe) or P(Very Poor or Severe) reaches the threshold.
    """
    bundle = app.state.bundle
    record = day.model_dump(exclude_none=True)           # only the fields that were sent
    city = find_city(day.city, bundle)
    if city is None:
        raise HTTPException(status_code=422, detail=f"unknown city {day.city!r}; known cities: {bundle['cities']}")

    if live:
        if len(record) > 1:
            raise HTTPException(status_code=422, detail='with live=true send only the city, e.g. {"city": "Delhi"}')
        based_on = live_data.latest_day()                # today once its bulletin is out, else yesterday
        try:
            record = live_data.build_record(city, based_on)
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f'could not reach CPCB: {e}')
        except RuntimeError as e:                        # too many missing bulletins, or no weather
            raise HTTPException(status_code=503, detail=str(e))
        try:
            check = check_forecast(city, based_on, bundle)
        except (requests.RequestException, RuntimeError) as e:   # the forecast does not depend on it
            check = {'date': based_on, 'error': str(e)}
        live_data.prune_bulletins(live_data.window_start(based_on - timedelta(days=1)))
    else:
        if len(record) == 1:
            raise HTTPException(status_code=422, detail='only the city was sent: add ?live=true for a '
                                                        'forecast from live data, or send all 27 fields')
        record['city'] = city
        try:
            check_record(record, bundle)                 # every field is there
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    result = predict(record, bundle)
    alert, rule = decide_alert(result, severe_threshold, dangerous_threshold)
    answer = {**result, 'alert': alert, 'alert_rule': rule}
    if live:
        answer.update(based_on=based_on, forecast_for=based_on + timedelta(days=1), check=check, input=record)
    return answer
