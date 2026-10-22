"""Step 1: Download and clean CPCB daily AQI bulletins (free, 2015-2025).

Source: Urban Emissions "AQI_bulletins" repository on GitHub, which compiles
the Central Pollution Control Board's (CPCB) daily AQI bulletins.
https://github.com/urbanemissionsinfo/AQI_bulletins  (GPL-3.0)

Input : downloaded automatically to data/raw/aqi_bulletins.csv
Output: data/processed/aqi_daily.csv (one row per city per day, 2015-2025)

Usage:  python prepare_aqi.py
"""
import difflib
from pathlib import Path

import pandas as pd
import requests

from cities import CITIES, END_DATE, START_DATE

SOURCE_URL = (
    "https://raw.githubusercontent.com/urbanemissionsinfo/AQI_bulletins/main/"
    "data/Processed/AllIndiaBulletinsMaster2025_openrefined.csv"
)
RAW = Path("data/raw/aqi_bulletins.csv")
OUT = Path("data/processed/aqi_daily.csv")

# Source column -> name used in the rest of the project
RENAME = {
    "no_stations": "number_of_monitoring_stations",
    "aqi_category": "air_quality_status",
    "aqi": "aqi_value",
    "prominent_pollutant": "prominent_pollutants",
}
KEEP_COLS = [
    "date", "city", "number_of_monitoring_stations",
    "prominent_pollutants", "aqi_value", "air_quality_status",
]


def download():
    if RAW.exists():
        print(f"{RAW} already exists, skipping download")
        return
    print(f"Downloading AQI bulletins (~19 MB) from GitHub...")
    r = requests.get(SOURCE_URL, timeout=300)
    r.raise_for_status()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(r.content)
    print(f"Saved {RAW}")


def main():
    download()

    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns=RENAME)
    print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]

    # Match bulletin city names to our city list (case-insensitive)
    df["city_raw"] = df["city"].astype(str).str.strip()
    lookup = {name.lower(): name for name in CITIES}
    df["city"] = df["city_raw"].str.lower().map(lookup)

    missing = sorted(set(CITIES) - set(df["city"].dropna()))
    if missing:
        names = df["city_raw"].unique().tolist()
        print("\nWARNING: these cities were not found in the file:")
        for name in missing:
            close = difflib.get_close_matches(name, names, n=3, cutoff=0.6)
            print(f"  {name!r}: closest names in file -> {close}")
        print("Edit cities.py if a closest name is the right one.\n")

    df = df.dropna(subset=["city"])
    df["aqi_value"] = pd.to_numeric(df["aqi_value"], errors="coerce")
    df["air_quality_status"] = df["air_quality_status"].astype(str).str.strip()
    df = df.dropna(subset=["date", "aqi_value"])

    df = (
        df[KEEP_COLS]
        .sort_values(["city", "date"])
        .drop_duplicates(["city", "date"], keep="last")
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    summary = df.groupby("city")["date"].agg(rows="count", first="min", last="max")
    print(summary.to_string())
    print(f"\nSaved {len(df):,} rows to {OUT}")


if __name__ == "__main__":
    main()