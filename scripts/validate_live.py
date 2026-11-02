"""Build the model's input for a city from real, recent data: the live mode of predict.py.

    uv run python scripts/predict.py --mode live --city Delhi

The data comes from the same sources as the training data:
  - AQI: the CPCB daily bulletin (a PDF per day, 4 PM, average of the past 24 hours), the source
    the training file was compiled from. Only the bulletins of the latest window are kept, in data/raw/bulletins/.
  - weather: the Open-Meteo archive API, with the same daily aggregation as code/download_weather.py

Needs the dev dependencies (requests, pdfplumber), so run it with `uv run`, not `uv run --no-dev`.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pdfplumber
import requests

from utility import ffill_aqi, add_trend_features

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))            # the data pipeline: city coordinates and weather
from cities import CITIES                                 # noqa: E402
from download_weather import fetch_hourly, to_daily       # noqa: E402

BULLETIN_URL = 'https://cpcb.gov.in/upload/Downloads/AQI_Bulletin_{:%Y%m%d}.pdf'
BULLETIN_DIR = PROJECT_DIR / 'data' / 'raw' / 'bulletins'
HISTORY_DAYS = 8          # aqi_dev7 compares the day with the mean of the 7 days before it
FILL_LIMIT = 3            # as in training: a missing day takes the last AQI, up to 3 days back
POLLUTANTS = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
SUBSCRIPTED = {'PM', 'NO', 'SO', 'O', 'NH'}              # written PM₁₀, NO₂, ...: the subscript is a separate line


def yesterday():
    return datetime.now(ZoneInfo('Asia/Kolkata')).date() - timedelta(days=1)


def download_bulletin(day):
    """The bulletin PDF for `day`, from the local copy if there is one. None if CPCB has none."""
    path = BULLETIN_DIR / f'AQI_Bulletin_{day:%Y%m%d}.pdf'
    if not path.exists():
        r = requests.get(BULLETIN_URL.format(day), timeout=60)
        if r.status_code != 200 or not r.content.startswith(b'%PDF'):
            return None
        BULLETIN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return path


def prune_bulletins(keep_from):
    """Delete the bulletins from before `keep_from`, so the folder holds only the latest window."""
    for path in BULLETIN_DIR.glob('AQI_Bulletin_*.pdf'):
        if path.stem.removeprefix('AQI_Bulletin_') < f'{keep_from:%Y%m%d}':
            path.unlink()


def parse_pollutants(cell):
    """'PM , PM\\n2.5 10' -> ['PM2.5', 'PM10']: the subscripts are on the second line, in order."""
    lines = (cell or '').split('\n')
    names = [n.strip() for n in lines[0].split(',') if n.strip()]
    subscripts = lines[1].split() if len(lines) > 1 else []
    return [n + subscripts.pop(0) if n in SUBSCRIPTED and subscripts else n for n in names]


def read_bulletin(path):
    """{city in lower case: (AQI, prominent pollutants)} for every city in the bulletin."""
    rows = {}
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    # Data rows: S.No, City, Air Quality, Index Value, Prominent Pollutant, Stations
                    if len(row) < 5 or not (row[0] or '').strip().isdigit():
                        continue
                    try:
                        aqi = float(row[3])
                    except (TypeError, ValueError):
                        continue
                    rows[' '.join(row[1].split()).lower()] = (aqi, parse_pollutants(row[4]))
    return rows


def bulletin_aqi(city, days):
    """One row per day: the city's AQI and prominent pollutants, NaN when the bulletin or the city is missing."""
    rows = []
    for day in days:
        path = download_bulletin(day)
        aqi, pollutants = read_bulletin(path).get(city.lower(), (None, [])) if path else (None, [])
        if aqi is None:
            print(f'  no AQI for {city} on {day}' + ('' if path else ': no bulletin'))
        rows.append({'city': city, 'date': pd.Timestamp(day), 'aqi_today': aqi, 'pollutants': pollutants})
    return pd.DataFrame(rows).astype({'aqi_today': float})


def window_start(day):
    """The first bulletin needed for `day`: the 8-day window, plus extra days so its first ones can be filled."""
    return day - timedelta(days=HISTORY_DAYS - 1 + FILL_LIMIT)


def build_record(city, day):
    """The model's input for `city` on `day`, built the same way as the training data."""
    history = bulletin_aqi(city, pd.date_range(window_start(day), day).date)

    # The same steps as prepare_dataset: fill short gaps (adds aqi_imputed), then the trend features
    history = add_trend_features(ffill_aqi(history, limit=FILL_LIMIT))
    today = history.iloc[-1]
    trend = ['aqi_today', 'aqi_roll7_mean', 'aqi_roll7_std', 'aqi_diff1', 'aqi_dev7']
    if today[trend].isna().any():
        raise RuntimeError(f'not enough CPCB data for {city} up to {day}: the bulletins miss too many days')

    lat, lon = CITIES[city]
    weather = to_daily(fetch_hourly(lat, lon, str(day), str(day))).iloc[0]

    record = {'city': city, 'month': int(today['month'])}
    for col in trend:
        record[col] = round(float(today[col]), 2)
    for col in ['temp_mean', 'temp_max', 'temp_min', 'temp_range', 'humidity_mean', 'dew_point_mean',
                'precip_sum', 'pressure_mean', 'cloud_cover_mean', 'wind_speed_mean', 'wind_speed_max',
                'wind_u_mean', 'wind_v_mean']:
        record[col] = float(weather[col])
    # As in training: flags from the day's own bulletin, so a filled day has none
    listed = {p.lower().replace('.', '') for p in today['pollutants']}
    for name in POLLUTANTS:
        record[f'has_{name}'] = int(name in listed)
    record['aqi_imputed'] = int(today['aqi_imputed'])
    return record


def actual_aqi(city, day):
    """The AQI CPCB reported for `day`, or None if the bulletin or the city is missing."""
    path = download_bulletin(day)
    aqi, _ = read_bulletin(path).get(city.lower(), (None, [])) if path else (None, [])
    return aqi
