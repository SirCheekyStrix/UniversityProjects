"""
fish_predict.py — predykcja brania ryb na 24h.

Użycie:
    python fish_predict.py
    python fish_predict.py --species szczupak
    python fish_predict.py --print
"""
import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

from fish_model import (
    SPECIES, FEATURE_COLS, MODEL_DIR,
    fetch_recent_weather, add_features
)

FORECAST_PATH = os.path.join(MODEL_DIR, "fish_forecast.json")


def load_models(species_list: list) -> dict:
    models = {}
    for species in species_list:
        path = os.path.join(MODEL_DIR, f"model_{species}.lgb")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Brak modelu dla '{species}'.\n"
                "Uruchom: python fish_train.py"
            )
        models[species] = lgb.Booster(model_file=path)
    return models


def predict_24h(species_filter: str = None, df_input=None) -> pd.DataFrame:
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Pobieranie danych pogodowych...")
    if df_input is not None and not isinstance(df_input, bool):
        df = add_features(df_input.copy())
    else:
        df = fetch_recent_weather()
        df = add_features(df)

    now     = pd.Timestamp.now().floor("h")
    end_24h = now + pd.Timedelta(hours=24)
    df      = df[(df["timestamp"] >= now) & (df["timestamp"] < end_24h)].copy()

    if len(df) == 0:
        raise ValueError("Brak danych dla przyszlych godzin")

    species_list = [species_filter] if species_filter else SPECIES
    models       = load_models(species_list)

    rows = []
    for _, row in df.iterrows():
        entry = {
            "timestamp":      str(row["timestamp"]),
            "hour":           int(row["timestamp"].hour),
            "temperature_c":  round(float(row["temperature_c"]), 1),
            "pressure_hpa":   round(float(row["pressure_hpa"]), 1),
            "d_pressure_3h":  round(float(row["d_pressure_3h"]), 2),
            "wind_speed_kmh": round(float(row["wind_speed_kmh"]), 1),
            "rain_rate_mm":   round(float(row["rain_rate_mm"]), 1),
            "rain_24h_sum":   round(float(row["rain_24h_sum"]), 1),
            "cloudcover_pct": round(float(row["cloudcover_pct"]), 0),
            "water_temp_c":   round(float(row["water_temp_c"]), 1),
            "water_temp_delta": round(float(row["water_temp_delta"]), 2),
            "moon_phase":     round(float(row["moon_phase"]), 2),
            "is_dawn":        int(row["is_dawn"]),
            "is_dusk":        int(row["is_dusk"]),
        }
        for species in species_list:
            X    = pd.DataFrame([row[FEATURE_COLS]])
            prob = float(models[species].predict(X)[0])
            entry[species] = round(float(np.clip(prob, 0, 100)), 1)
        rows.append(entry)

    return pd.DataFrame(rows)


def print_forecast(df: pd.DataFrame):
    species_cols = [c for c in SPECIES if c in df.columns]

    # Drukuj w dwóch grupach żeby zmieścić 10 gatunków
    groups = [species_cols[:5], species_cols[5:]]

    for group in groups:
        if not group:
            continue
        header = f"  {'Godzina':<18}" + "".join(f"{s.capitalize():>11}" for s in group)
        sep    = "─" * len(header)
        print(f"\n{sep}\n{header}\n{sep}")

        for _, row in df.iterrows():
            ts   = row["timestamp"][11:16]
            dawn = " 🌅" if row["is_dawn"] else ""
            dusk = " 🌆" if row["is_dusk"] else ""
            rain = " 🌧" if row["rain_rate_mm"] > 0.5 else ""
            line = f"  {ts}{dawn}{dusk}{rain:<13}"
            for s in group:
                v    = row[s]
                icon = "🟢" if v >= 70 else ("🟡" if v >= 40 else "🔴")
                line += f"{icon}{v:>4.0f}%   "
            print(line)

    print(f"\n  🟢 >=70% doskonale | 🟡 40-69% dobre | 🔴 <40% slabe")

    print(f"\n{'─'*50}")
    print("  NAJLEPSZA GODZINA NA POLOW:")
    for s in species_cols:
        best     = df.loc[df[s].idxmax()]
        time_str = best["timestamp"][11:16]
        val      = best[s]
        temp     = best["temperature_c"]
        dp       = best["d_pressure_3h"]
        trend    = "↓" if dp < -0.5 else ("↑" if dp > 0.5 else "→")
        print(f"  {s.capitalize():<10}  {time_str}  ->  {val:.0f}%"
              f"  (woda: {best['water_temp_c']:.1f}°C, cisn: {trend})")


def save_forecast_json(df: pd.DataFrame):
    species_cols = [c for c in SPECIES if c in df.columns]
    result = {
        "generated_at": datetime.now().isoformat(),
        "model":        "LightGBM",
        "species":      species_cols,
        "forecast":     df.to_dict(orient="records"),
        "best_hours": {
            s: {
                "timestamp":   str(df.loc[df[s].idxmax(), "timestamp"]),
                "probability": float(df[s].max()),
            }
            for s in species_cols
        }
    }
    with open(FORECAST_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"JSON zapisany: {FORECAST_PATH}")

    try:
        from influx_writer import write_fish_24h
        n = write_fish_24h(result)
        print(f"  -> InfluxDB predictions: {n} punktow")
    except Exception as e:
        print(f"  InfluxDB pominieto: {e}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--print",   action="store_true", dest="print_out")
    args = parser.parse_args()

    forecast_df = predict_24h(species_filter=args.species)
    save_forecast_json(forecast_df)

    if args.print_out:
        print_forecast(forecast_df)
    else:
        species_cols = [c for c in SPECIES if c in forecast_df.columns]
        print("Najlepsze godziny:")
        for s in species_cols:
            best = forecast_df.loc[forecast_df[s].idxmax()]
            print(f"  {s.capitalize():<10} {best['timestamp'][11:16]}  {best[s]:.0f}%")