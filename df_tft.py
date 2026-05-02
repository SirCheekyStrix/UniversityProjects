import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PARQUET_FILE = DATA_DIR / "station_timeseries_tft.parquet"

def fetch_open_meteo(lat=53.0138, lon=18.5981, start="2024-02-24", end="2026-02-24"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "precipitation"
        ],
        "timezone": "UTC"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()["hourly"]

    df = pd.DataFrame(data)
    # konwersja timestamp na datetime
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])

    # rename kolumn do TFT
    df = df.rename(columns={
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "surface_pressure": "pressure",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "wind_gusts_10m": "wind_gust",
        "precipitation": "rain_rate"
    })

    # dodaj stację, jeśli jest jedna
    df["station_id"] = 0

    # sortowanie po czasie
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Tworzymy ciągły time_idx potrzebny do TFT
    df = df.drop_duplicates(subset="timestamp")
    df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600
    df["time_idx"] = df["time_idx"].astype(int)

    return df

def save_data(df):
    df.to_parquet(PARQUET_FILE, index=False)
    print(f"Zapisano dane do {PARQUET_FILE}")
    print(df.head())
    print(df.tail())

def main():
    # pobierz dane 2 lata (np. 2024–2025)
    df = fetch_open_meteo(start="2024-02-24", end="2026-02-24")
    print(f"Liczba rekordów: {len(df)}")
    print(f"Zakres godzin: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    save_data(df)

if __name__ == "__main__":
    main()