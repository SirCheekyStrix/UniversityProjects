"""
tft_model.py — wspólna biblioteka dla systemu TFT 10-dniowej prognozy pogody.

Kluczowa różnica v2: dane NWP z Open-Meteo Forecast API jako known_future.
Model widzi prognozę synoptyczną na 16 dni → nie musi samodzielnie wymyślać frontu.
"""
import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from datetime import date, timedelta

LAT, LON      = 53.0138, 18.5981
LOCATION_ID   = "torun"

MODEL_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tft_model_files")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODEL_DIR, "tft_weather.ckpt")
DATA_PATH     = os.path.join(MODEL_DIR, "training_data.parquet")
FORECAST_PATH = os.path.join(MODEL_DIR, "tft_forecast.json")

ENCODER_LEN   = 60
PRED_LEN      = 10
MIN_OBS       = ENCODER_LEN + PRED_LEN

TARGETS = [
    "temp_max",
    "temp_min",
    "precip_sum",
    "wind_max",
    "wind_mean",
    "pressure_mean",
    "cloudcover_mean",
    "humidity_mean",
]

# NWP jako known_future — kluczowe dla jakości prognozy
KNOWN_FUTURE_REALS = [
    "doy_sin", "doy_cos",
    "nwp_temp_max",
    "nwp_temp_min",
    "nwp_precip",
    "nwp_pressure",
    "nwp_wind_max",
    "nwp_cloudcover",
]
KNOWN_FUTURE_CATS = ["month", "weekday"]

OBSERVED_PAST = [
    "temp_mean",
    "d_pressure_3d",
    "precip_7d_sum",
    "temp_anomaly",
    "nwp_error_temp",
]


def _parse_hourly_means(h: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "time":      pd.to_datetime(h["time"]),
        "pressure":  pd.array(h["pressure_msl"],        dtype=float),
        "humidity":  pd.array(h["relativehumidity_2m"],  dtype=float),
        "windspeed": pd.array(h["windspeed_10m"],        dtype=float),
    })
    df["date"] = df["time"].dt.normalize()
    return df.groupby("date").agg(
        pressure_mean=("pressure",  "mean"),
        humidity_mean=("humidity",  "mean"),
        wind_mean    =("windspeed", "mean"),
    ).reset_index()


def _build_df(d: dict, daily_means: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "date":            pd.to_datetime(d["time"]),
        "temp_max":        pd.array(d["temperature_2m_max"], dtype=float),
        "temp_min":        pd.array(d["temperature_2m_min"], dtype=float),
        "temp_mean":       [(a+b)/2 if a is not None and b is not None else 0.0 for a,b in zip(d["temperature_2m_max"], d["temperature_2m_min"])],
        "precip_sum":      pd.array(d["precipitation_sum"],  dtype=float),
        "wind_max":        pd.array(d["windspeed_10m_max"],  dtype=float),
        "cloudcover_mean": pd.array(d["cloudcover_mean"],    dtype=float),
    })
    df = df.merge(daily_means, on="date", how="left")
    df["wind_mean"]     = df["wind_mean"].fillna(df["wind_max"] * 0.6)
    df["pressure_mean"] = df["pressure_mean"].fillna(1013.0)
    df["humidity_mean"] = df["humidity_mean"].fillna(75.0)
    # Dla archiwum NWP = obserwacje (idealna prognoza — uczy struktury)
    df["nwp_temp_max"]   = df["temp_max"]
    df["nwp_temp_min"]   = df["temp_min"]
    df["nwp_precip"]     = df["precip_sum"]
    df["nwp_pressure"]   = df["pressure_mean"]
    df["nwp_wind_max"]   = df["wind_max"]
    df["nwp_cloudcover"] = df["cloudcover_mean"]
    return df


def _fetch_daily_archive(start_date: str, end_date: str) -> pd.DataFrame:
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,windspeed_10m_max,cloudcover_mean"
        f"&hourly=pressure_msl,relativehumidity_2m,windspeed_10m"
        f"&start_date={start_date}&end_date={end_date}&timezone=UTC"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    return _build_df(data["daily"], _parse_hourly_means(data["hourly"]))


def _fetch_daily_forecast_with_nwp(past_days: int = 65, forecast_days: int = 16) -> pd.DataFrame:
    """Pobiera historię + prognozę NWP. Dla przyszłych dni nwp_* = prawdziwa prognoza GFS/ECMWF."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,windspeed_10m_max,cloudcover_mean"
        f"&hourly=pressure_msl,relativehumidity_2m,windspeed_10m"
        f"&past_days={past_days}&forecast_days={forecast_days}&timezone=UTC"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    return _build_df(data["daily"], _parse_hourly_means(data["hourly"]))


def fetch_training_data(years_back: int = 5) -> pd.DataFrame:
    today  = date.today()
    frames = []
    for year in range(today.year - years_back, today.year):
        print(f"  Rok {year}...", end=" ", flush=True)
        chunk = _fetch_daily_archive(f"{year}-01-01", f"{year}-12-31")
        frames.append(chunk)
        print(f"{len(chunk)} dni")
    end_recent = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"  {today.year} do {end_recent}...", end=" ", flush=True)
    chunk = _fetch_daily_archive(f"{today.year}-01-01", end_recent)
    frames.append(chunk)
    print(f"{len(chunk)} dni")
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_recent_data() -> pd.DataFrame:
    return _fetch_daily_forecast_with_nwp(past_days=65, forecast_days=16)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    num_cols = [c for c in df.columns if c != "date" and df[c].dtype != object]
    df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()

    doy = df["date"].dt.day_of_year
    df["doy_sin"]  = np.sin(2 * np.pi * doy / 365).round(4)
    df["doy_cos"]  = np.cos(2 * np.pi * doy / 365).round(4)
    df["month"]    = df["date"].dt.month.astype(str)
    df["weekday"]  = df["date"].dt.weekday.astype(str)

    df["d_pressure_3d"]  = df["pressure_mean"].diff(3).fillna(0).round(2)
    df["precip_7d_sum"]  = df["precip_sum"].rolling(7, min_periods=1).sum().round(2)
    df["temp_30d_mean"]  = df["temp_mean"].rolling(30, min_periods=7).mean()
    df["temp_anomaly"]   = (df["temp_mean"] - df["temp_30d_mean"]).fillna(0).round(2)
    df["nwp_error_temp"] = (df["temp_max"] - df["nwp_temp_max"].shift(1)).fillna(0).round(2)

    df["precip_sum"]      = df["precip_sum"].clip(lower=0)
    df["nwp_precip"]      = df["nwp_precip"].clip(lower=0)
    df["wind_max"]        = df["wind_max"].clip(lower=0)
    df["wind_mean"]       = df["wind_mean"].clip(lower=0)
    df["cloudcover_mean"] = df["cloudcover_mean"].clip(0, 100)
    df["nwp_cloudcover"]  = df["nwp_cloudcover"].clip(0, 100)
    df["humidity_mean"]   = df["humidity_mean"].clip(0, 100)

    df["time_idx"] = (df["date"] - df["date"].min()).dt.days
    df["group_id"] = LOCATION_ID
    return df


def compute_pop(row) -> float:
    precip   = float(row.get("precip_sum", 0) or 0)
    humidity = float(row.get("humidity_mean", 50) or 50)
    cloud    = float(row.get("cloudcover_mean", 50) or 50)
    nwp_p    = float(row.get("nwp_precip", precip) or precip)
    p_eff    = max(precip, nwp_p)
    c        = 1 - math.exp(-p_eff / 2.0)
    a        = (humidity / 100) * (0.5 + 0.5 * (cloud / 100))
    return round(min(100, c * a * 100 + (p_eff > 0.3) * 15), 1)