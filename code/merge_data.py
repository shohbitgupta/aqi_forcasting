"""Step 3: Merge AQI bulletins with daily weather into one modelling dataset.

Each city gets a complete daily calendar, so days without an AQI bulletin
appear as empty rows instead of disappearing. This keeps lag features honest
later on (a "1-day lag" really is yesterday, not the last day with data).

Input : data/processed/aqi_daily.csv, data/processed/weather_daily.csv
Output: data/processed/aqi_weather_2015_2025.csv

Usage:  python merge_data.py
"""
from pathlib import Path

import pandas as pd

AQI = Path("data/processed/aqi_daily.csv")
WEATHER = Path("data/processed/weather_daily.csv")
OUT = Path("data/processed/aqi_weather_2015_2025.csv")


def full_calendar(aqi):
    """One row per city per day between that city's first and last bulletin."""
    grids = []
    for city, g in aqi.groupby("city"):
        days = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        grids.append(pd.DataFrame({"city": city, "date": days}))
    return pd.concat(grids, ignore_index=True)


def split_name(date):
    if date.year <= 2022:
        return "train"
    if date.year == 2023:
        return "val"
    return "test"


def main():
    aqi = pd.read_csv(AQI, parse_dates=["date"])
    weather = pd.read_csv(WEATHER, parse_dates=["date"])

    df = full_calendar(aqi)
    df = df.merge(aqi, on=["city", "date"], how="left")
    df = df.merge(weather, on=["city", "date"], how="left")
    df["split"] = df["date"].apply(split_name)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Saved {len(df):,} rows x {df.shape[1]} columns to {OUT}\n")
    print("Days with no AQI bulletin:", df["aqi_value"].isna().sum())
    print("Days with no weather    :", df["temp_mean"].isna().sum())
    print("\nRows per split:")
    print(df["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("\nAQI category distribution:")
    print(df["air_quality_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
