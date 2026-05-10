"""
fish_model.py — model LightGBM do predykcji prawdopodobieństwa brania ryb.

Gatunki: karp, leszcz, szczupak, okon, sandacz, ploc, lin, sum, klen, wzdrega

Dane treningowe:
  - Syntetyczne bite_probability oparte na biologii gatunków
  - Fine-tuning wzorców z pliku Fitzroy Basin (temperatura, turbidity, opady)
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
import lightgbm as lgb
import joblib
from datetime import date, datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

LAT, LON  = 53.0138, 18.5981
MODEL_DIR = os.path.join(os.path.dirname(__file__), "fish_model_files")
os.makedirs(MODEL_DIR, exist_ok=True)

SPECIES = ["karp", "leszcz", "szczupak", "okon", "sandacz",
           "ploc", "lin", "sum", "klen", "wzdrega", "amur"]

FEATURE_COLS = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "d_pressure_3h",
    "d_pressure_24h",
    "wind_speed_kmh",
    "wind_gust_kmh",
    "rain_rate_mm",
    "rain_24h_sum",       # suma opadów z ostatnich 24h — wzorzec Fitzroy
    "cloudcover_pct",
    "water_temp_c",
    "water_temp_delta",   # zmiana temp wody 24h — aktywność po schłodzeniu
    "moon_phase",
    "moon_illumination",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "is_dawn",
    "is_dusk",
    "pressure_trend",
]

# ─────────────────────────────────────────────
# BIOLOGICZNE PREFERENCJE GATUNKÓW
# Zaktualizowane na podstawie literatury + korelacji z Fitzroy Basin
# ─────────────────────────────────────────────
SPECIES_PREFS = {
    "karp": {
        "water_temp_opt": (18, 25), "water_temp_min": 10, "water_temp_max": 30,
        "pressure_pref": "stable",
        "active_hours": (5, 21), "dawn_dusk_bonus": 0.20,
        "rain_penalty": 0.05,    # lubi lekki deszcz / po deszczu
        "wind_max": 20,          "moon_pref": "full",
        "cloud_pref": "any",
        "temp_drop_bonus": 0.15, # aktywny przy spadku temp wody (Fitzroy)
        "post_rain_bonus": 0.15, # aktywny dzień po deszczu
    },
    "leszcz": {
        "water_temp_opt": (15, 22), "water_temp_min": 8, "water_temp_max": 28,
        "pressure_pref": "stable",
        "active_hours": (4, 10), "dawn_dusk_bonus": 0.35,
        "rain_penalty": 0.05,    "wind_max": 15,
        "moon_pref": "new",      "cloud_pref": "cloudy",
        "temp_drop_bonus": 0.10, "post_rain_bonus": 0.10,
    },
    "szczupak": {
        "water_temp_opt": (10, 18), "water_temp_min": 4, "water_temp_max": 24,
        "pressure_pref": "falling",
        "active_hours": (6, 20), "dawn_dusk_bonus": 0.40,
        "rain_penalty": -0.10,   "wind_max": 30,
        "moon_pref": "any",      "cloud_pref": "cloudy",
        "temp_drop_bonus": 0.20, "post_rain_bonus": 0.05,
    },
    "okon": {
        "water_temp_opt": (16, 22), "water_temp_min": 8, "water_temp_max": 26,
        "pressure_pref": "any",
        "active_hours": (6, 20), "dawn_dusk_bonus": 0.30,
        "rain_penalty": 0.05,    "wind_max": 25,
        "moon_pref": "any",      "cloud_pref": "any",
        "temp_drop_bonus": 0.10, "post_rain_bonus": 0.08,
    },
    "sandacz": {
        "water_temp_opt": (14, 20), "water_temp_min": 6, "water_temp_max": 24,
        "pressure_pref": "falling",
        "active_hours": (18, 23), "dawn_dusk_bonus": 0.50,
        "rain_penalty": -0.05,   "wind_max": 20,
        "moon_pref": "new",      "cloud_pref": "cloudy",
        "temp_drop_bonus": 0.15, "post_rain_bonus": 0.10,
    },
    "ploc": {
        # Płoć — aktywna przez większość dnia, lubi ciepłe płytkie wody
        "water_temp_opt": (16, 24), "water_temp_min": 6, "water_temp_max": 28,
        "pressure_pref": "stable",
        "active_hours": (6, 20),  "dawn_dusk_bonus": 0.20,
        "rain_penalty": 0.05,     "wind_max": 20,
        "moon_pref": "any",       "cloud_pref": "any",
        "temp_drop_bonus": 0.05,  "post_rain_bonus": 0.10,
    },
    "lin": {
        # Lin — lubi ciepłe, spokojne, muliste wody; aktywny głównie rano
        "water_temp_opt": (18, 26), "water_temp_min": 10, "water_temp_max": 30,
        "pressure_pref": "stable",
        "active_hours": (5, 11),  "dawn_dusk_bonus": 0.35,
        "rain_penalty": 0.05,     "wind_max": 10,
        "moon_pref": "full",      "cloud_pref": "cloudy",
        "temp_drop_bonus": 0.05,  "post_rain_bonus": 0.05,
    },
    "sum": {
        # Sum — nocny drapieżnik, lubi ciepło i burzę
        "water_temp_opt": (20, 28), "water_temp_min": 12, "water_temp_max": 32,
        "pressure_pref": "falling",
        "active_hours": (20, 6),  "dawn_dusk_bonus": 0.35,
        "rain_penalty": -0.15,    "wind_max": 35,   # bardzo aktywny przy burzy
        "moon_pref": "new",       "cloud_pref": "cloudy",
        "temp_drop_bonus": 0.05,  "post_rain_bonus": 0.20,
    },
    "klen": {
        # Kleń — ryba rzek, aktywna w chłodniejszej wodzie, lubi prąd
        "water_temp_opt": (12, 20), "water_temp_min": 5, "water_temp_max": 25,
        "pressure_pref": "any",
        "active_hours": (6, 20),  "dawn_dusk_bonus": 0.25,
        "rain_penalty": -0.05,    "wind_max": 25,
        "moon_pref": "any",       "cloud_pref": "any",
        "temp_drop_bonus": 0.15,  "post_rain_bonus": 0.15,
    },
    "amur": {
        # Amur — roślinożerca, lubi ciepłą wodę, aktywny w dzień
        "water_temp_opt": (20, 28), "water_temp_min": 12, "water_temp_max": 32,
        "pressure_pref": "stable",
        "active_hours": (7, 20),  "dawn_dusk_bonus": 0.15,
        "rain_penalty": 0.05,     "wind_max": 15,
        "moon_pref": "full",      "cloud_pref": "clear",
        "temp_drop_bonus": 0.05,  "post_rain_bonus": 0.05,
    },
    "wzdrega": {
        # Wzdręga — aktywna przy powierzchni, lubi spokojną pogodę
        "water_temp_opt": (14, 22), "water_temp_min": 6, "water_temp_max": 26,
        "pressure_pref": "stable",
        "active_hours": (7, 19),  "dawn_dusk_bonus": 0.15,
        "rain_penalty": 0.10,     "wind_max": 12,   # bardzo wrażliwa na wiatr
        "moon_pref": "any",       "cloud_pref": "clear",
        "temp_drop_bonus": 0.05,  "post_rain_bonus": 0.05,
    },
}

# ─────────────────────────────────────────────
# POBIERANIE DANYCH
# ─────────────────────────────────────────────

def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relativehumidity_2m,pressure_msl,"
        f"windspeed_10m,windgusts_10m,precipitation,cloudcover"
        f"&start_date={start_date}&end_date={end_date}&timezone=UTC"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({
        "timestamp":      pd.to_datetime(h["time"]),
        "temperature_c":  h["temperature_2m"],
        "humidity_pct":   h["relativehumidity_2m"],
        "pressure_hpa":   h["pressure_msl"],
        "wind_speed_kmh": h["windspeed_10m"],
        "wind_gust_kmh":  h["windgusts_10m"],
        "rain_rate_mm":   h["precipitation"],
        "cloudcover_pct": h["cloudcover"],
    })


def fetch_recent_weather() -> pd.DataFrame:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relativehumidity_2m,pressure_msl,"
        f"windspeed_10m,windgusts_10m,precipitation,cloudcover"
        f"&past_days=3&forecast_days=2&timezone=UTC"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({
        "timestamp":      pd.to_datetime(h["time"]),
        "temperature_c":  h["temperature_2m"],
        "humidity_pct":   h["relativehumidity_2m"],
        "pressure_hpa":   h["pressure_msl"],
        "wind_speed_kmh": h["windspeed_10m"],
        "wind_gust_kmh":  h["windgusts_10m"],
        "rain_rate_mm":   h["precipitation"],
        "cloudcover_pct": h["cloudcover"],
    })

# ─────────────────────────────────────────────
# INŻYNIERIA CECH
# ─────────────────────────────────────────────

def moon_phase(dt: pd.Timestamp) -> tuple:
    known_new = datetime(2000, 1, 6)
    days  = (dt.to_pydatetime().replace(tzinfo=None) - known_new).days
    phase = (days % 29.53058867) / 29.53058867
    illum = 50 * (1 - math.cos(2 * math.pi * phase))
    return round(phase, 3), round(illum, 1)


def solar_elevation(dt: pd.Timestamp, lat: float) -> float:
    doy        = dt.day_of_year
    hour_utc   = dt.hour + dt.minute / 60
    decl       = math.radians(23.45 * math.sin(math.radians(360/365 * (doy - 81))))
    hour_angle = math.radians(15 * (hour_utc - 12))
    lat_r      = math.radians(lat)
    sin_elev   = (math.sin(lat_r)*math.sin(decl) +
                  math.cos(lat_r)*math.cos(decl)*math.cos(hour_angle))
    return math.degrees(math.asin(max(-1, min(1, sin_elev))))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = ["temperature_c", "humidity_pct", "pressure_hpa",
                "wind_speed_kmh", "wind_gust_kmh", "rain_rate_mm", "cloudcover_pct"]
    df[num_cols] = df[num_cols].ffill().bfill()

    df["d_pressure_3h"]  = df["pressure_hpa"].diff(3).fillna(0)
    df["d_pressure_24h"] = df["pressure_hpa"].diff(24).fillna(0)
    df["pressure_trend"] = np.sign(df["d_pressure_3h"]).astype(int)

    # Suma opadów z ostatnich 24h — wzorzec Fitzroy (aktywność po deszczu)
    df["rain_24h_sum"] = df["rain_rate_mm"].rolling(24, min_periods=1).sum()

    h   = df["timestamp"].dt.hour
    doy = df["timestamp"].dt.day_of_year
    df["hour_sin"] = np.sin(2 * np.pi * h   / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h   / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * doy / 365)

    elevations = [solar_elevation(ts, LAT) for ts in df["timestamp"]]
    df["solar_elev"] = elevations
    df["is_dawn"] = ((df["solar_elev"] > -6) & (df["solar_elev"] < 6) &
                     (df["timestamp"].dt.hour < 12)).astype(int)
    df["is_dusk"] = ((df["solar_elev"] > -6) & (df["solar_elev"] < 6) &
                     (df["timestamp"].dt.hour >= 12)).astype(int)

    moon_data = [moon_phase(ts) for ts in df["timestamp"]]
    df["moon_phase"]        = [m[0] for m in moon_data]
    df["moon_illumination"] = [m[1] for m in moon_data]

    # Temperatura wody — rolling 72h + lag 6h
    df["water_temp_c"] = (df["temperature_c"]
                          .rolling(72, min_periods=1).mean()
                          .shift(6)
                          .fillna(df["temperature_c"]))

    # Zmiana temp wody w ciągu 24h — kluczowy sygnał z Fitzroy
    df["water_temp_delta"] = df["water_temp_c"].diff(24).fillna(0)

    return df


# ─────────────────────────────────────────────
# GENERATOR SYNTETYCZNEGO BITE_PROBABILITY
# ─────────────────────────────────────────────

def compute_bite_probability(row: pd.Series, species: str) -> float:
    p     = SPECIES_PREFS[species]
    score = 0.50

    # 1. Temperatura wody
    wt = row["water_temp_c"]
    opt_lo, opt_hi = p["water_temp_opt"]
    if wt < p["water_temp_min"] or wt > p["water_temp_max"]:
        return np.random.uniform(0, 5)
    if opt_lo <= wt <= opt_hi:
        score += 0.20
    else:
        dist   = min(abs(wt - opt_lo), abs(wt - opt_hi))
        score -= 0.02 * dist

    # 2. Wzorzec Fitzroy: spadek temp wody = aktywność
    delta = row.get("water_temp_delta", 0)
    if delta < -0.5:   # temp wody spada — ryby aktywne
        score += p.get("temp_drop_bonus", 0.10)
    elif delta > 1.5:  # gwałtowne ocieplenie = pasywność
        score -= 0.10

    # 3. Wzorzec Fitzroy: aktywność PO deszczu (nie podczas)
    rain_now  = row["rain_rate_mm"]
    rain_24h  = row.get("rain_24h_sum", 0)
    if rain_now < 0.5 and rain_24h > 3:   # było > 3mm w ostatnich 24h, teraz sucho
        score += p.get("post_rain_bonus", 0.10)
    if rain_now > 5:   # ulewa = źle
        score -= 0.30
    elif rain_now > 0:
        score += p["rain_penalty"]

    # 4. Ciśnienie
    dp3 = row["d_pressure_3h"]
    if p["pressure_pref"] == "falling" and dp3 < -1.5:
        score += 0.20
    elif p["pressure_pref"] == "rising"  and dp3 > 1.5:
        score += 0.15
    elif p["pressure_pref"] == "stable"  and abs(dp3) < 1.0:
        score += 0.15
    elif p["pressure_pref"] == "falling" and dp3 > 1.5:
        score -= 0.20
    if abs(dp3) > 4:
        score -= 0.25

    # 5. Pora dnia — sum ma godziny nocne (20-6)
    hour  = row["timestamp"].hour if hasattr(row["timestamp"], "hour") else 12
    h_lo, h_hi = p["active_hours"]
    if h_lo > h_hi:   # nocny gatunek (np. sum: 20-6)
        in_active = hour >= h_lo or hour <= h_hi
    else:
        in_active = h_lo <= hour <= h_hi
    score += 0.10 if in_active else -0.20

    # 6. Świt / zmierzch
    if row["is_dawn"] or row["is_dusk"]:
        score += p["dawn_dusk_bonus"]

    # 7. Wiatr
    wind = row["wind_speed_kmh"]
    if wind > p["wind_max"]:
        score -= 0.20 * (wind - p["wind_max"]) / 10

    # 8. Księżyc
    illum = row["moon_illumination"]
    if p["moon_pref"] == "full" and illum > 75:
        score += 0.10
    elif p["moon_pref"] == "new" and illum < 25:
        score += 0.10

    # 9. Zachmurzenie
    cloud = row["cloudcover_pct"]
    if p["cloud_pref"] == "cloudy" and cloud > 60:
        score += 0.08
    elif p["cloud_pref"] == "clear" and cloud < 30:
        score += 0.08

    score += np.random.normal(0, 0.07)
    return round(float(np.clip(score * 100, 0, 100)), 1)


# ─────────────────────────────────────────────
# FINE-TUNING Z FITZROY BASIN
# ─────────────────────────────────────────────

def load_fitzroy_features(csv_path: str) -> pd.DataFrame:
    """
    Wczytuje dane Fitzroy Basin i tworzy cechy do fine-tuningu.
    Mapuje australijskie warunki na polskie gatunki przez normalizację temperatury.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["Number of fish"] = pd.to_numeric(df["Number of fish"], errors="coerce").fillna(0)

    # Parsuj datę
    df["timestamp"] = pd.to_datetime(df["Date"], format="%H:%M:%S %d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Normalizuj temperaturę wody: australijska 26°C ≈ polska 18°C (inne optimum)
    # Skalowanie liniowe: [21, 29] → [13, 22]
    wt_raw = df["Water Temp (Deg. C)"]
    df["water_temp_c"] = 13 + (wt_raw - 21) / (29 - 21) * (22 - 13)

    # Zmiana temp wody 24h (daily data = diff(1))
    df["water_temp_delta"] = df["water_temp_c"].diff(1).fillna(0)

    # Pozostałe cechy
    df["rain_rate_mm"]   = df["Rainfall (mm)"].clip(0)
    df["rain_24h_sum"]   = df["rain_rate_mm"].rolling(3, min_periods=1).sum()
    df["turbidity"]      = df["Turbidity (NTU)"]
    df["conductivity"]   = df["E Conduct. (us/cm)"]

    # Bite probability z liczby ryb (normalizuj do 0-100)
    max_fish = df["Number of fish"].quantile(0.95)
    df["bite_probability"] = (df["Number of fish"] / max(max_fish, 1) * 100).clip(0, 100)

    # Podstawowe cechy pogodowe (brakuje — wypełnij typowymi wartościami dla Australii)
    df["temperature_c"]  = df["water_temp_c"] + 3   # przybliżenie
    df["humidity_pct"]   = 70.0
    df["pressure_hpa"]   = 1013.0
    df["d_pressure_3h"]  = 0.0
    df["d_pressure_24h"] = 0.0
    df["wind_speed_kmh"] = 10.0
    df["wind_gust_kmh"]  = 15.0
    df["cloudcover_pct"] = 50.0
    df["pressure_trend"] = 0
    df["moon_phase"]     = 0.25
    df["moon_illumination"] = 50.0

    # Czasowe
    h   = df["timestamp"].dt.hour
    doy = df["timestamp"].dt.day_of_year
    df["hour_sin"] = np.sin(2 * np.pi * h   / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h   / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * doy / 365)
    df["is_dawn"]  = 0
    df["is_dusk"]  = 0

    return df[FEATURE_COLS + ["bite_probability"]].dropna()


# ─────────────────────────────────────────────
# TRENING
# ─────────────────────────────────────────────

def generate_training_data(years_back: int = 3) -> pd.DataFrame:
    today  = date.today()
    frames = []
    for year in range(today.year - years_back, today.year + 1):
        start = f"{year}-01-01"
        end   = min(f"{year}-12-31", str(today - timedelta(days=7)))
        if start > end:
            continue
        print(f"  Pobieram {year}...", end=" ", flush=True)
        try:
            chunk = fetch_weather(start, end)
            frames.append(chunk)
            print(f"{len(chunk)}h")
        except Exception as e:
            print(f"BLAD: {e}")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = add_features(df)

    all_dfs = []
    for species in SPECIES:
        df_s = df.copy()
        df_s["species"] = species
        df_s["bite_probability"] = df_s.apply(
            lambda row: compute_bite_probability(row, species), axis=1
        )
        all_dfs.append(df_s)

    result = pd.concat(all_dfs, ignore_index=True)
    print(f"  Lacznie: {len(result)} rekordow ({len(df)}h x {len(SPECIES)} gatunkow)")
    return result


def train(df: pd.DataFrame, species: str,
          fitzroy_df: pd.DataFrame = None) -> lgb.Booster:
    df_s = df[df["species"] == species].dropna(subset=FEATURE_COLS + ["bite_probability"])
    X    = df_s[FEATURE_COLS]
    y    = df_s["bite_probability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Fine-tuning: dodaj dane Fitzroy z wagą 3x
    if fitzroy_df is not None and len(fitzroy_df) > 0:
        X_fitz = fitzroy_df[FEATURE_COLS]
        y_fitz = fitzroy_df["bite_probability"]
        X_train = pd.concat([X_train] + [X_fitz] * 3, ignore_index=True)
        y_train = pd.concat([y_train] + [y_fitz] * 3, ignore_index=True)
        print(f"  [{species}] + {len(X_fitz)*3} Fitzroy samples")

    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "learning_rate":    0.03,
        "num_leaves":       31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "reg_alpha":        0.1,
        "reg_lambda":       0.2,
        "verbose":          -1,
    }

    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_test, label=y_test)],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(200),
        ],
    )

    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print(f"  [{species:>8}] RMSE: {rmse:.1f}% | train={len(X_train)} val={len(X_test)}")
    return model


def train_all(years_back: int = 3, fitzroy_csv: str = None):
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] === TRENING MODELI RYB ===")
    print(f"Gatunki: {', '.join(SPECIES)}")

    # Fine-tuning data
    fitzroy_df = None
    if fitzroy_csv and os.path.exists(fitzroy_csv):
        print(f"Wczytuje dane Fitzroy Basin: {fitzroy_csv}")
        fitzroy_df = load_fitzroy_features(fitzroy_csv)
        print(f"  Fitzroy: {len(fitzroy_df)} rekordow, "
              f"dni z polowem: {(fitzroy_df['bite_probability'] > 10).sum()}")

    print(f"Generuje dane treningowe ({years_back} lata)...")
    df = generate_training_data(years_back)

    print("\nTrenuje modele...")
    rmse_results = {}
    for species in SPECIES:
        model = train(df, species, fitzroy_df)
        path  = os.path.join(MODEL_DIR, f"model_{species}.lgb")
        model.save_model(path)
        preds = model.predict(df[df["species"] == species][FEATURE_COLS])
        true  = df[df["species"] == species]["bite_probability"].values
        rmse_results[species] = round(float(np.sqrt(mean_squared_error(true, preds))), 2)

    meta = {
        "trained_at":    datetime.now().isoformat(),
        "species":       SPECIES,
        "features":      FEATURE_COLS,
        "records":       len(df),
        "years_back":    years_back,
        "fitzroy_used":  fitzroy_csv is not None,
        "rmse":          rmse_results,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nModele zapisane: {MODEL_DIR}")
    print(f"RMSE: {rmse_results}")
    return rmse_results


# ─────────────────────────────────────────────
# PREDYKCJA
# ─────────────────────────────────────────────

def load_models() -> dict:
    models = {}
    for species in SPECIES:
        path = os.path.join(MODEL_DIR, f"model_{species}.lgb")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Brak modelu dla {species}. Uruchom: python fish_train.py"
            )
        models[species] = lgb.Booster(model_file=path)
    return models


def predict_now(species_filter: str = None) -> pd.DataFrame:
    print(f"[{datetime.now():%H:%M}] Pobieranie danych pogodowych...")
    df = fetch_recent_weather()
    df = add_features(df)

    now     = pd.Timestamp.now().floor("h")
    end_24h = now + pd.Timedelta(hours=24)
    df      = df[(df["timestamp"] >= now) & (df["timestamp"] < end_24h)]

    if len(df) == 0:
        raise ValueError("Brak danych dla przyszlych godzin")

    models       = load_models()
    species_list = [species_filter] if species_filter else SPECIES

    results = []
    for _, row in df.iterrows():
        entry = {"timestamp": str(row["timestamp"]), "hour": row["timestamp"].hour}
        for species in species_list:
            X    = pd.DataFrame([row[FEATURE_COLS]])
            prob = float(models[species].predict(X)[0])
            entry[species] = round(float(np.clip(prob, 0, 100)), 1)
        results.append(entry)

    return pd.DataFrame(results)


def print_forecast(df: pd.DataFrame):
    species_cols = [c for c in df.columns if c in SPECIES]
    header = f"{'Godzina':<20}" + "".join(f"{s.capitalize():>12}" for s in species_cols)
    print(f"\n{'='*len(header)}\n{header}\n{'='*len(header)}")
    for _, row in df.iterrows():
        ts   = row["timestamp"][:16]
        vals = ""
        for s in species_cols:
            v = row[s]
            icon = "🟢" if v >= 70 else ("🟡" if v >= 40 else "🔴")
            vals += f"{icon}{v:>5.0f}%    "
        print(f"  {ts}  {vals}")
    print(f"\n🟢 >=70% doskonale | 🟡 40-69% dobre | 🔴 <40% slabe")
    print(f"\nNAJLEPSZA GODZINA:")
    for s in species_cols:
        best = df.loc[df[s].idxmax()]
        print(f"  {s.capitalize():<10} {best['timestamp'][11:16]}  ->  {best[s]:.0f}%")


def save_forecast_json(df: pd.DataFrame):
    path = os.path.join(MODEL_DIR, "fish_forecast.json")
    species_cols = [c for c in SPECIES if c in df.columns]
    result = {
        "generated_at": datetime.now().isoformat(),
        "location":     {"lat": LAT, "lon": LON},
        "species":      species_cols,
        "forecast":     df.to_dict(orient="records"),
        "best_hours": {
            s: {"timestamp": str(df.loc[df[s].idxmax(), "timestamp"]),
                "probability": float(df[s].max())}
            for s in species_cols
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"JSON zapisany: {path}")
    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--years",   type=int, default=3)
    parser.add_argument("--fitzroy", type=str, default=None,
                        help="Sciezka do pliku CSV Fitzroy Basin")
    args = parser.parse_args()

    if args.train:
        train_all(years_back=args.years, fitzroy_csv=args.fitzroy)
    if args.predict:
        forecast_df = predict_now(species_filter=args.species)
        print_forecast(forecast_df)
        save_forecast_json(forecast_df)
    if not args.train and not args.predict:
        parser.print_help()