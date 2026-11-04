"""Daily weather on the CPCB bulletin's window (4 PM to 4 PM) instead of the calendar day.

The bulletin published at 4 PM on day D averages the 24 hours from 4 PM on D-1 to 4 PM on D.
The calendar-day weather in weather_daily.csv covers midnight to midnight, so its last 8 hours
fall in the NEXT bulletin's window, the one the model forecasts. This script aggregates the same
Open-Meteo hourly data over each bulletin's own window, for experiments/weather_window.ipynb.

Output: data/processed/weather_bulletin_window.csv            (2015-2025, one row per city and bulletin date)
        data/processed/weather_bulletin_window_live_check.csv (1 Oct - 3 Nov 2026, around the live check days)
Cache : data/raw/weather_hourly/<city>.csv                     (hourly data, not in git)

Usage:  python code/download_weather_window.py                                   # all 14 cities
        python code/download_weather_window.py Delhi Noida "Greater Noida"          # only these cities
"""
import sys
import time
from pathlib import Path

import pandas as pd

from cities import CITIES, END_DATE, START_DATE
from download_weather import fetch_hourly, to_daily

CACHE_DIR = Path("data/raw/weather_hourly")
OUT = Path("data/processed/weather_bulletin_window.csv")
OUT_LIVE = Path("data/processed/weather_bulletin_window_live_check.csv")
LIVE_START, LIVE_END = "2026-10-01", "2026-11-03"
SHIFT = pd.Timedelta(hours=8)      # 4 PM on D-1 + 8 h = midnight on D: each window becomes one "day"


def hourly(city, lat, lon):
    """Hourly weather from the evening before START_DATE to END_DATE, cached per city."""
    cache = CACHE_DIR / f"{city.replace(' ', '_').lower()}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    first = pd.Timestamp(START_DATE) - pd.Timedelta(days=1)     # the first window starts at 4 PM the day before
    parts = []
    for year in range(first.year, int(END_DATE[:4]) + 1):
        start = max(f"{year}-01-01", f"{first:%Y-%m-%d}")
        end = min(f"{year}-12-31", END_DATE)
        print(f"{city}: {start} -> {end}")
        parts.append(fetch_hourly(lat, lon, start, end))
        time.sleep(2)                                            # be polite to the free API
    h = pd.concat(parts, ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h.to_csv(cache, index=False)
    return h


def bulletin_window(h):
    """Daily values per bulletin date: the hours from 4 PM the day before to 3 PM that day (24 hours)."""
    h = h.copy()
    h["time"] = pd.to_datetime(h["time"]) + SHIFT
    full = h.groupby(h["time"].dt.normalize())["time"].transform("size") == 24
    return to_daily(h[full])                                    # incomplete windows (the ends) are dropped


def main(cities):
    unknown = set(cities) - set(CITIES)
    if unknown:
        sys.exit(f"unknown cities: {sorted(unknown)}; choose from {list(CITIES)}")
    train, live = [], []
    for city in cities:
        lat, lon = CITIES[city]
        d = bulletin_window(hourly(city, lat, lon))
        train.append(d.assign(city=city))
        start = f"{pd.Timestamp(LIVE_START) - pd.Timedelta(days=1):%Y-%m-%d}"
        d = bulletin_window(fetch_hourly(lat, lon, start, LIVE_END))
        live.append(d.assign(city=city))
        time.sleep(1)
    for frames, path in [(train, OUT), (live, OUT_LIVE)]:
        out = pd.concat(frames, ignore_index=True)
        out = out[["city"] + [c for c in out.columns if c != "city"]]
        out.to_csv(path, index=False)
        print(f"Saved {len(out):,} rows to {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(CITIES))
