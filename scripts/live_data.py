"""Build the model's input for a city from real, recent data: live forecasts.

    python scripts/forecast.py --mode live --city Delhi         # command line
    POST /predict?live=true  {"city": "Delhi"}                  # the web service

The data comes from the same sources as the training data:
  - AQI: the CPCB daily bulletin (a PDF per day, 4 PM, average of the past 24 hours), the source
    the training file was compiled from. Only the bulletins of the latest window are kept, in data/raw/bulletins/
    (or in $AQI_BULLETIN_DIR, for a server where the project folder is read-only).
  - weather: the Open-Meteo archive API, with the same daily aggregation as code/download_weather.py
"""
import os
import re
import sys
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pypdfium2
import requests

from utility import ffill_aqi, add_trend_features

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))            # the data pipeline: city coordinates and weather
from cities import CITIES                                 # noqa: E402
from download_weather import fetch_hourly, to_daily       # noqa: E402

BULLETIN_URL = 'https://cpcb.gov.in/upload/Downloads/AQI_Bulletin_{:%Y%m%d}.pdf'
BULLETIN_DIR = Path(os.environ.get('AQI_BULLETIN_DIR', PROJECT_DIR / 'data' / 'raw' / 'bulletins'))
HISTORY_DAYS = 8          # aqi_dev7 compares the day with the mean of the 7 days before it
FILL_LIMIT = 3            # as in training: a missing day takes the last AQI, up to 3 days back
POLLUTANTS = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
# A row of the bulletin's table: S.No, City, Air Quality, Index Value, Prominent Pollutants, Stations
BULLETIN_ROW = re.compile(r'\d+ (.+?) (?:Good|Satisfactory|Moderate|Poor|Very Poor|Severe) '
                          r'(\d+(?:\.\d+)?) (.*?) ?\d+/\d+')


RECHECK_S = 600          # while today's bulletin is not out, ask CPCB again at most every 10 minutes
_last_miss = {}          # day -> when CPCB last had no bulletin for it


def latest_day():
    """The newest day with a bulletin: today once CPCB has published it (about 4 PM IST), else yesterday.

    A forecast from it is for the next day, so after about 4 PM it is for tomorrow."""
    today = datetime.now(ZoneInfo('Asia/Kolkata')).date()
    if time.monotonic() - _last_miss.get(today, -RECHECK_S) >= RECHECK_S:
        try:
            if download_bulletin(today):
                return today
        except requests.RequestException:
            pass                         # CPCB not reachable now: use yesterday's, which is usually kept
        _last_miss[today] = time.monotonic()
    return today - timedelta(days=1)


def download_bulletin(day):
    """The bulletin PDF for `day`, from the local copy if there is one. None if CPCB has none."""
    path = BULLETIN_DIR / f'AQI_Bulletin_{day:%Y%m%d}.pdf'
    if not path.exists():
        r = requests.get(BULLETIN_URL.format(day), timeout=(15, 60))     # connect, read
        if r.status_code != 200 or not r.content.startswith(b'%PDF'):
            return None
        BULLETIN_DIR.mkdir(parents=True, exist_ok=True)
        # Write, then rename: a request reading the file at the same moment never sees half of it
        partial = path.with_suffix(f'.{os.getpid()}.{id(r)}.part')
        partial.write_bytes(r.content)
        partial.replace(path)
    return path


def prune_bulletins(keep_from):
    """Delete the bulletins from before `keep_from`, so the folder holds only the latest window."""
    for path in BULLETIN_DIR.glob('AQI_Bulletin_*.pdf'):
        if path.stem.removeprefix('AQI_Bulletin_') < f'{keep_from:%Y%m%d}':
            path.unlink()


@lru_cache(maxsize=16)
def read_bulletin(path):
    """{city in lower case: (AQI, prominent pollutants)} for every city in the bulletin.

    It reads the PDF's text, a few hundredths of a second per bulletin, rather than its tables, which
    is 30 times slower: that matters on a small server. A day's bulletin never changes, so the web
    service also keeps each one in memory once read.
    """
    pdf = pypdfium2.PdfDocument(path)
    try:
        lines = [line for page in pdf for line in page.get_textpage().get_text_range().splitlines()]
    finally:
        pdf.close()
    # A row can be split over several lines, e.g. a long city name or a pollutant after O3:
    # join lines from one that starts with the row number until they make a whole row
    rows, parts = {}, []
    for line in lines:
        line = ' '.join(line.split())
        if not parts and not line[:1].isdigit():
            continue                                   # a heading, not part of a row
        parts.append(line)
        if match := BULLETIN_ROW.fullmatch(' '.join(parts)):
            city, aqi, pollutants = match.groups()
            rows[city.lower()] = (float(aqi), [p.strip() for p in pollutants.split(',') if p.strip()])
            parts = []
        elif len(parts) > 12:                          # not a row after all
            parts = []
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
    weather = to_daily(fetch_hourly(lat, lon, str(day), str(day), retries=2)).iloc[0]

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
