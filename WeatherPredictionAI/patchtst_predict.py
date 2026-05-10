"""
patchtst_predict.py — generuje 24h godzinową prognozę pogody z modelu PatchTST
z blendingiem NWP (Open-Meteo Forecast API).
"""
import argparse
import json
import math
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from datetime import datetime

from patchtst_model import (
    LAT, LON, FORECAST_PATH,
    TARGETS, OUTPUT_HOURS,
    fetch_recent_window, add_features, predict,
    _fetch_with_retry,
)


# ─────────────────────────────────────────────
# NWP BLEND
# ─────────────────────────────────────────────

# Mapowanie TARGETS → kolumna NWP z Forecast API
NWP_FIELD = {
    "temperature":   "temperature_2m",
    "apparent_temp": "apparent_temperature",
    "dewpoint":      "dewpoint_2m",
    "humidity":      "relativehumidity_2m",
    "pressure":      "pressure_msl",
    "wind_speed":    "windspeed_10m",
    "wind_gust":     "windgusts_10m",
    "rain_rate":     "precipitation",
    "cloudcover":    "cloudcover",
}

# Wagi NWP wg godziny prognozy (h=0 to następna godzina, h=23 to za 24h)
# Im bliżej — tym bardziej ufamy NWP
NWP_WEIGHTS = {
    "temperature":   [0.90, 0.88, 0.85, 0.82, 0.78, 0.74, 0.70, 0.65,
                      0.60, 0.55, 0.50, 0.45, 0.40, 0.36, 0.32, 0.28,
                      0.25, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10],
    "apparent_temp": [0.88, 0.85, 0.82, 0.79, 0.75, 0.71, 0.67, 0.62,
                      0.57, 0.52, 0.47, 0.42, 0.38, 0.34, 0.30, 0.26,
                      0.23, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08],
    "rain_rate":     [0.95, 0.93, 0.90, 0.87, 0.84, 0.80, 0.76, 0.72,
                      0.68, 0.64, 0.60, 0.56, 0.52, 0.48, 0.44, 0.40,
                      0.36, 0.32, 0.28, 0.24, 0.20, 0.16, 0.12, 0.08],
    "wind_speed":    [0.85, 0.82, 0.79, 0.76, 0.72, 0.68, 0.64, 0.60,
                      0.56, 0.52, 0.48, 0.44, 0.40, 0.36, 0.32, 0.28,
                      0.25, 0.22, 0.19, 0.16, 0.14, 0.12, 0.10, 0.08],
    "pressure":      [0.88, 0.85, 0.82, 0.79, 0.76, 0.72, 0.68, 0.64,
                      0.60, 0.56, 0.52, 0.48, 0.44, 0.40, 0.36, 0.32,
                      0.28, 0.25, 0.22, 0.19, 0.16, 0.14, 0.12, 0.10],
}
# Domyślne wagi dla pozostałych zmiennych
_DEFAULT_WEIGHTS = [0.80, 0.77, 0.74, 0.71, 0.68, 0.64, 0.60, 0.56,
                    0.52, 0.48, 0.44, 0.40, 0.36, 0.32, 0.28, 0.24,
                    0.21, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05]


def fetch_nwp_forecast(lat=LAT, lon=LON) -> dict:
    """Pobiera 24h prognozy NWP z Open-Meteo Forecast API."""
    fields = ",".join(NWP_FIELD.values())
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={fields}"
        f"&forecast_days=2&timezone=UTC"
    )
    try:
        data = _fetch_with_retry(url)
        h    = data["hourly"]
        now_utc = pd.Timestamp.utcnow().floor("h").tz_localize(None)
        times   = pd.to_datetime(h["time"])
        # Tylko przyszłe godziny (od now+1h do now+24h)
        mask    = (times > now_utc) & (times <= now_utc + pd.Timedelta(hours=OUTPUT_HOURS))
        result  = {}
        for target, api_field in NWP_FIELD.items():
            vals = np.array(h[api_field], dtype=float)
            result[target] = vals[mask]
        return result
    except Exception as e:
        print(f"  ⚠ NWP fetch blad: {e}")
        return {}


def blend_with_nwp(result_df: pd.DataFrame, nwp: dict) -> pd.DataFrame:
    """Blend wyników PatchTST z NWP wg wag godzinowych."""
    df = result_df.copy()
    n  = min(len(df), OUTPUT_HOURS)

    for target in TARGETS:
        if target not in nwp or len(nwp[target]) < n:
            continue

        weights = NWP_WEIGHTS.get(target, _DEFAULT_WEIGHTS)
        patchtst_vals = df[target].values[:n].copy()
        nwp_vals      = nwp[target][:n]

        blended = np.array([
            weights[i] * nwp_vals[i] + (1 - weights[i]) * patchtst_vals[i]
            for i in range(n)
        ])

        # Fizyczne ograniczenia
        if target == "rain_rate":
            blended = np.where(blended < 0.1, 0.0, blended)
        if target in ("wind_speed", "wind_gust"):
            blended = np.maximum(0, blended)
        if target == "humidity":
            blended = np.clip(blended, 0, 100)
        if target == "cloudcover":
            blended = np.clip(blended, 0, 100)

        df.loc[df.index[:n], target] = blended

    # Punkt rosy <= temperatura
    if "dewpoint" in df.columns and "temperature" in df.columns:
        df["dewpoint"] = np.minimum(df["dewpoint"].values, df["temperature"].values - 0.1)

    return df


# ─────────────────────────────────────────────
# GŁÓWNA FUNKCJA
# ─────────────────────────────────────────────

def run_prediction(df_input=None, lat=None, lon=None):
    """Zwraca DataFrame z prognozą 24h (PatchTST + NWP blend)."""
    _lat = lat or LAT
    _lon = lon or LON

    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Pobieram dane...")
    if df_input is not None:
        df = add_features(df_input.copy())
    else:
        raw = fetch_recent_window()
        df  = add_features(raw)

    print("Generuję prognozę PatchTST...")
    result_df = predict(df)

    print("Pobieram NWP i blenduje...")
    nwp = fetch_nwp_forecast(_lat, _lon)
    if nwp:
        result_df = blend_with_nwp(result_df, nwp)
        print(f"  Blend OK — {len(nwp)} zmiennych")
    else:
        print("  Blend pominięty — brak NWP")

    # Zapisz JSON
    forecast = {
        "generated_at": datetime.now().isoformat(),
        "model":        "PatchTST+NWP",
        "hours":        OUTPUT_HOURS,
        "forecast":     [],
    }
    for _, row in result_df.iterrows():
        entry = {"timestamp": str(row["timestamp"])}
        for col in TARGETS + ["uv_index", "pop_pct"]:
            if col in row.index:
                entry[col] = round(float(row[col]), 2)
        forecast["forecast"].append(entry)

    os.makedirs(os.path.dirname(FORECAST_PATH), exist_ok=True)
    with open(FORECAST_PATH, "w") as f:
        json.dump(forecast, f, indent=2)
    print(f"Prognoza zapisana: {FORECAST_PATH}")

    # Zapis do InfluxDB
    try:
        from influx_writer import write_weather_24h
        n = write_weather_24h(forecast)
        print(f"  → InfluxDB: {n} punktów")
    except Exception as e:
        print(f"  ⚠ InfluxDB pominięty: {e}")

    return result_df


def print_forecast(df):
    temp      = df["temperature"].values
    print(f"\n  Temp:       min {temp.min():+.1f}°C / max {temp.max():+.1f}°C / śr {temp.mean():.1f}°C")
    print(f"  Odczuwalna: min {df['apparent_temp'].min():+.1f}°C / max {df['apparent_temp'].max():+.1f}°C")
    print(f"  Wiatr:      max {df['wind_speed'].max():.1f} km/h | Porywy: max {df['wind_gust'].max():.1f} km/h")
    print(f"  Ciśnienie śr: {df['pressure'].mean():.1f} hPa")
    print(f"  Wilgotność: śr {df['humidity'].mean():.1f}% | Zachmurzenie: śr {df['cloudcover'].mean():.1f}%")
    print(f"  UV max: {df['uv_index'].max():.1f} | Opady max: {df['rain_rate'].max():.1f} mm/h")
    pop = df["pop_pct"]
    print(f"  PoP max: {pop.max():.1f}% | śr: {pop.mean():.1f}% | godzin ≥50%: {(pop>=50).sum()}")

    header = (f"\n{'Godz':<22}"
              f"{'Temp':>7}{'Odcz':>7}{'Wiatr':>8}{'Porywy':>8}"
              f"{'Ciśn':>10}{'Opady':>8}{'RH':>6}{'Rosy':>6}"
              f"{'Chmury':>8}{'UV':>6}")
    print(header)
    print("-" * 94)

    for _, row in df.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        print(
            f"  {ts:%Y-%m-%d %H:%M}"
            f"  {row['temperature']:>+6.1f}°C"
            f"  {row['apparent_temp']:>+6.1f}°C"
            f"  {row['wind_speed']:>5.1f}km/h"
            f"  {row['wind_gust']:>6.1f}km/h"
            f"  {row['pressure']:>8.1f}hPa"
            f"  {row['rain_rate']:>5.1f}mm/h"
            f"  {row['humidity']:>4.0f}%RH"
            f"  {row['dewpoint']:>+5.1f}°C"
            f"  {row['cloudcover']:>5.0f}%"
            f"  UV:{row['uv_index']:.1f}"
            f"  PoP:{row['pop_pct']:>4.0f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", dest="print_out")
    args = parser.parse_args()

    result_df = run_prediction()

    if args.print_out:
        print_forecast(result_df)
    else:
        temp = result_df["temperature"]
        print(f"\n  Temp: {temp.min():+.1f}°C / {temp.max():+.1f}°C"
              f"  |  Wiatr max: {result_df['wind_speed'].max():.1f} km/h"
              f"  |  PoP max: {result_df['pop_pct'].max():.0f}%")