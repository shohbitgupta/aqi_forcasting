"""Step 2: Download historical weather for each city from Open-Meteo (free, no API key).

Downloads hourly weather one year at a time, aggregates it to daily values,
and caches one file per city so the script can be re-run safely if it stops.

Output: data/raw/weather/<city>.csv          (cache, one per city)
        data/processed/weather_daily.csv     (all cities combined)

Usage:  python download_weather.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from cities import CITIES, END_DATE, START_DATE

URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
]
CACHE_DIR = Path("data/raw/weather")
OUT = Path("data/processed/weather_daily.csv")


def fetch_hourly(lat, lon, start, end, retries=5):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "ms",
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(URL, params=params, timeout=120)
            if r.status_code == 429:  # rate limited: wait and retry
                wait = 60 * attempt
                print(f"    rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return pd.DataFrame(r.json()["hourly"])
        except requests.RequestException as e:
            print(f"    attempt {attempt} failed: {e}")
            time.sleep(10 * attempt)
    raise RuntimeError(f"Could not download {start}..{end} for ({lat}, {lon})")


def to_daily(hourly):
    h = hourly.copy()
    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.normalize()

    # Average wind direction properly using u/v components (a plain mean of
    # angles is wrong, e.g. the mean of 350 and 10 degrees should be 0, not 180).
    rad = np.deg2rad(h["wind_direction_10m"])
    h["wind_u"] = -h["wind_speed_10m"] * np.sin(rad)
    h["wind_v"] = -h["wind_speed_10m"] * np.cos(rad)

    d = h.groupby("date").agg(
        temp_mean=("temperature_2m", "mean"),
        temp_max=("temperature_2m", "max"),
        temp_min=("temperature_2m", "min"),
        humidity_mean=("relative_humidity_2m", "mean"),
        dew_point_mean=("dew_point_2m", "mean"),
        precip_sum=("precipitation", "sum"),
        pressure_mean=("surface_pressure", "mean"),
        cloud_cover_mean=("cloud_cover", "mean"),
        wind_speed_mean=("wind_speed_10m", "mean"),
        wind_speed_max=("wind_speed_10m", "max"),
        wind_u_mean=("wind_u", "mean"),
        wind_v_mean=("wind_v", "mean"),
    )
    d["wind_dir_mean"] = (
        np.rad2deg(np.arctan2(-d["wind_u_mean"], -d["wind_v_mean"])) + 360
    ) % 360
    d["temp_range"] = d["temp_max"] - d["temp_min"]
    return d.round(2).reset_index()


def download_city(city, lat, lon):
    cache = CACHE_DIR / f"{city.replace(' ', '_').lower()}.csv"
    if cache.exists():
        print(f"{city}: cached, skipping")
        return pd.read_csv(cache, parse_dates=["date"])

    start_year, end_year = int(START_DATE[:4]), int(END_DATE[:4])
    parts = []
    for year in range(start_year, end_year + 1):
        start = max(f"{year}-01-01", START_DATE)
        end = min(f"{year}-12-31", END_DATE)
        print(f"{city}: {start} -> {end}")
        parts.append(fetch_hourly(lat, lon, start, end))
        time.sleep(2)  # be polite to the free API

    daily = to_daily(pd.concat(parts, ignore_index=True))
    daily.insert(0, "city", city)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache, index=False)
    return daily


def main():
    frames = [download_city(c, lat, lon) for c, (lat, lon) in CITIES.items()]
    weather = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(OUT, index=False)
    print(f"\nSaved {len(weather):,} rows to {OUT}")


if __name__ == "__main__":
    main()
