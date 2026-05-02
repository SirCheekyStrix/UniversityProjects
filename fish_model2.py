"""
fish_model.py — model LightGBM do predykcji prawdopodobieństwa brania ryb.
Dane treningowe generowane syntetycznie na podstawie danych Open-Meteo
i biologicznych preferencji gatunków.

Użycie:
    python fish_model.py --train       # wygeneruj dane i wytrenuj
    python fish_model.py --predict     # predykcja na teraz
    python fish_model.py --predict --species szczupak
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

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────
LAT, LON   = 53.0138, 18.5981   # Toruń — zmień na swoje

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "fish_model_files")
os.makedirs(MODEL_DIR, exist_ok=True)

SPECIES = ["karp", "leszcz", "szczupak", "okon", "sandacz"]

FEATURE_COLS = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "d_pressure_3h",     # zmiana ciśnienia — kluczowe dla aktywności
    "d_pressure_24h",
    "wind_speed_kmh",
    "wind_gust_kmh",
    "rain_rate_mm",
    "cloudcover_pct",
    "water_temp_c",      # szacowana z temp powietrza
    "moon_phase",        # 0=nów, 0.5=pełnia, 1=nów
    "moon_illumination",
    "hour_sin",          # pora dnia zakodowana cyklicznie
    "hour_cos",
    "doy_sin",           # pora roku
    "doy_cos",
    "is_dawn",           # 1 jeśli świt (30min przed/po wschodzie)
    "is_dusk",           # 1 jeśli zmierzch
    "pressure_trend",    # -1 spada, 0 stabilne, +1 rośnie
]

# ─────────────────────────────────────────────
# BIOLOGICZNE PREFERENCJE GATUNKÓW
# Każdy słownik definiuje optymalne warunki dla danego gatunku
# ─────────────────────────────────────────────
SPECIES_PREFS = {
    "karp": {
        "water_temp_opt": (18, 25),    # °C optimum
        "water_temp_min": 10,
        "water_temp_max": 30,
        "pressure_pref": "stable",     # stable/rising/falling
        "active_hours": (5, 21),       # aktywny 5:00-21:00
        "dawn_dusk_bonus": 0.2,        # bonus za świt/zmierzch
        "rain_penalty": 0.1,           # lubi lekki deszcz
        "wind_max": 20,                # do 20 km/h
        "moon_pref": "full",           # pełnia
        "cloud_pref": "any",
    },
    "leszcz": {
        "water_temp_opt": (15, 22),
        "water_temp_min": 8,
        "water_temp_max": 28,
        "pressure_pref": "stable",
        "active_hours": (4, 10),       # głównie rano
        "dawn_dusk_bonus": 0.35,
        "rain_penalty": 0.05,
        "wind_max": 15,
        "moon_pref": "new",            # nów
        "cloud_pref": "cloudy",
    },
    "szczupak": {
        "water_temp_opt": (10, 18),
        "water_temp_min": 4,
        "water_temp_max": 24,
        "pressure_pref": "falling",    # aktywny przy spadającym ciśnieniu
        "active_hours": (6, 20),
        "dawn_dusk_bonus": 0.4,
        "rain_penalty": -0.1,          # lubi deszcz (negatywna wartość = bonus)
        "wind_max": 30,
        "moon_pref": "any",
        "cloud_pref": "cloudy",
    },
    "okon": {
        "water_temp_opt": (16, 22),
        "water_temp_min": 8,
        "water_temp_max": 26,
        "pressure_pref": "any",
        "active_hours": (6, 20),
        "dawn_dusk_bonus": 0.3,
        "rain_penalty": 0.05,
        "wind_max": 25,
        "moon_pref": "any",
        "cloud_pref": "any",
    },
    "sandacz": {
        "water_temp_opt": (14, 20),
        "water_temp_min": 6,
        "water_temp_max": 24,
        "pressure_pref": "falling",
        "active_hours": (18, 23),      # głównie wieczór i noc
        "dawn_dusk_bonus": 0.5,
        "rain_penalty": -0.05,
        "wind_max": 20,
        "moon_pref": "new",            # ciemność = aktywny
        "cloud_pref": "cloudy",
    },
}

FILE_NAME_AU = "Fitzroy_Basin.csv"

# ─────────────────────────────────────────────
# POBIERANIE DANYCH Z OPEN-METEO
# ─────────────────────────────────────────────

def load_au_data(file_path):
    """
    Wczytuje dane z rzeki Dee (Fitzroy Basin) z uwzględnieniem 
    specyficznych nazw kolumn z jednostkami.
    """
    if not os.path.exists(file_path):
        print(f" Błąd: Plik {file_path} nie istnieje!")
        return None
    
    try:
        # Wczytanie pliku CSV
        df = pd.read_csv(file_path, sep=',', on_bad_lines='skip', low_memory=False)
        
        # 1. Mapowanie kolumn dokładnie tak, jak w Twoim pliku
        mapping = {
            'Date': 'date',
            'Rainfall (mm)': 'rainfall',
            'Level (Metres)': 'level',
            'E Conduct. (us/cm)': 'conductivity',
            'Water Temp (Deg. C)': 'water_temp',
            'pH (pH units)': 'ph',
            'Turbidity (NTU)': 'turbidity',
            'Number of fish': 'fish_count'
        }
        
        # Zmiana nazw tylko tych kolumn, które faktycznie są w pliku
        existing_mapping = {k: v for k, v in mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mapping)
        
        # 2. Konwersja daty (uwzględniamy format: 00:00:00 01/01/2010)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date'])
        
        # 3. Upewnienie się, że mamy wszystkie zmienne dla modelu (jeśli brak, dajemy 0)
        required_cols = ['rainfall', 'level', 'conductivity', 'water_temp', 'ph', 'turbidity', 'fish_count']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
            # Konwersja na liczby (usuwa ewentualne teksty/błędy)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # 4. Inżynieria cech specyficzna dla rzeki (np. trend poziomu wody)
        df = df.sort_values('date')
        df['level_diff'] = df['level'].diff().fillna(0)
        
        # 5. Estymacja tlenu (Dissolved Oxygen) - kluczowe dla modelu
        # Wzór biologiczny oparty na temperaturze wody
        df['est_do'] = 14.6 - 0.45 * df['water_temp'] + 0.008 * (df['water_temp']**2)
        
        print(f" [OK] Wczytano {len(df)} rekordów z {file_path}")
        return df

    except Exception as e:
        print(f" Błąd krytyczny podczas wczytywania CSV: {e}")
        return None
    

def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Pobiera dane historyczne z Open-Meteo Archive API."""
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
    df = pd.DataFrame({
        "timestamp":      pd.to_datetime(h["time"]),
        "temperature_c":  h["temperature_2m"],
        "humidity_pct":   h["relativehumidity_2m"],
        "pressure_hpa":   h["pressure_msl"],
        "wind_speed_kmh": h["windspeed_10m"],
        "wind_gust_kmh":  h["windgusts_10m"],
        "rain_rate_mm":   h["precipitation"],
        "cloudcover_pct": h["cloudcover"],
    })
    return df

def fetch_recent_weather() -> pd.DataFrame:
    """Pobiera ostatnie 3 dni (kontekst dla d_pressure_24h) + 2 dni prognozy."""
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
    """Przybliżona faza księżyca (0=nów, 0.5=pełnia)."""
    known_new = datetime(2000, 1, 6)
    days = (dt.to_pydatetime().replace(tzinfo=None) - known_new).days
    cycle = 29.53058867
    phase = (days % cycle) / cycle
    illumination = 50 * (1 - math.cos(2 * math.pi * phase))
    return round(phase, 3), round(illumination, 1)

def solar_elevation(dt: pd.Timestamp, lat: float) -> float:
    """Elewacja słońca w stopniach."""
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

    # Interpoluj NaN
    num_cols = ["temperature_c", "humidity_pct", "pressure_hpa",
                "wind_speed_kmh", "wind_gust_kmh", "rain_rate_mm", "cloudcover_pct"]
    df[num_cols] = df[num_cols].ffill().bfill()

    # Ciśnienie — pochodne i trend
    df["d_pressure_3h"]  = df["pressure_hpa"].diff(3).fillna(0)
    df["d_pressure_24h"] = df["pressure_hpa"].diff(24).fillna(0)
    df["pressure_trend"] = np.sign(df["d_pressure_3h"]).astype(int)

    # Czas
    h   = df["timestamp"].dt.hour
    doy = df["timestamp"].dt.day_of_year
    df["hour_sin"] = np.sin(2 * np.pi * h   / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h   / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * doy / 365)

    # Świt / zmierzch
    elevations = [solar_elevation(ts, LAT) for ts in df["timestamp"]]
    df["solar_elev"] = elevations
    df["is_dawn"] = ((df["solar_elev"] > -6) & (df["solar_elev"] < 6) &
                     (df["timestamp"].dt.hour < 12)).astype(int)
    df["is_dusk"] = ((df["solar_elev"] > -6) & (df["solar_elev"] < 6) &
                     (df["timestamp"].dt.hour >= 12)).astype(int)

    # Księżyc
    moon_data = [moon_phase(ts) for ts in df["timestamp"]]
    df["moon_phase"]        = [m[0] for m in moon_data]
    df["moon_illumination"] = [m[1] for m in moon_data]

    # Temperatura wody — szacowana z temp powietrza (opóźniona o 6h, wygładzona)
    df["water_temp_c"] = (df["temperature_c"]
                          .rolling(72, min_periods=1).mean()
                          .shift(6)
                          .fillna(df["temperature_c"]))

    return df

# ─────────────────────────────────────────────
# GENERATOR SYNTETYCZNEGO BITE_PROBABILITY
# ─────────────────────────────────────────────

def compute_bite_probability(row: pd.Series, species: str) -> float:
    """
    Wylicza syntetyczne prawdopodobieństwo brania (0-100)
    na podstawie biologicznych preferencji gatunku.
    """
    p = SPECIES_PREFS[species]
    score = 0.5  # startujemy od 50%

    # 1. Temperatura wody
    wt       = row["water_temp_c"]
    opt_lo, opt_hi = p["water_temp_opt"]
    if wt < p["water_temp_min"] or wt > p["water_temp_max"]:
        return np.random.uniform(0, 5)   # poza zakresem = prawie 0
    if opt_lo <= wt <= opt_hi:
        score += 0.20
    else:
        dist = min(abs(wt - opt_lo), abs(wt - opt_hi))
        score -= 0.02 * dist

    # 2. Ciśnienie
    dp3 = row["d_pressure_3h"]
    if p["pressure_pref"] == "falling" and dp3 < -1.5:
        score += 0.20
    elif p["pressure_pref"] == "rising" and dp3 > 1.5:
        score += 0.15
    elif p["pressure_pref"] == "stable" and abs(dp3) < 1.0:
        score += 0.15
    elif p["pressure_pref"] == "falling" and dp3 > 1.5:
        score -= 0.20   # aktywny przy spadającym, pasywny przy rosnącym
    elif abs(dp3) > 4:
        score -= 0.25   # gwałtowna zmiana = brak aktywności dla wszystkich

    # 3. Pora dnia
    hour = row["timestamp"].hour if hasattr(row["timestamp"], "hour") else row.get("hour", 12)
    h_lo, h_hi = p["active_hours"]
    if h_lo <= hour <= h_hi:
        score += 0.10
    else:
        score -= 0.20

    # 4. Świt i zmierzch
    if row["is_dawn"] or row["is_dusk"]:
        score += p["dawn_dusk_bonus"]

    # 5. Deszcz
    rain = row["rain_rate_mm"]
    if rain > 0:
        score += p["rain_penalty"]   # może być bonus (ujemna kara) lub kara
    if rain > 5:
        score -= 0.30   # ulewa = źle dla wszystkich

    # 6. Wiatr
    wind = row["wind_speed_kmh"]
    if wind > p["wind_max"]:
        score -= 0.20 * (wind - p["wind_max"]) / 10

    # 7. Księżyc
    moon = row["moon_phase"]
    illum = row["moon_illumination"]
    if p["moon_pref"] == "full" and illum > 75:
        score += 0.10
    elif p["moon_pref"] == "new" and illum < 25:
        score += 0.10

    # 8. Zachmurzenie
    cloud = row["cloudcover_pct"]
    if p["cloud_pref"] == "cloudy" and cloud > 60:
        score += 0.08
    elif p["cloud_pref"] == "clear" and cloud < 30:
        score += 0.08

    # Dodaj szum ± 10% żeby unikać idealnych wartości
    score += np.random.normal(0, 0.08)

    # Konwertuj do 0-100%
    prob = np.clip(score * 100, 0, 100)
    return round(prob, 1)

def generate_training_data(years_back: int = 3) -> pd.DataFrame:
    """Pobiera dane z Open-Meteo i generuje syntetyczne bite_probability dla każdego gatunku."""
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
            print(f"BŁĄD: {e}")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = add_features(df)

    # Generuj bite_probability dla każdego gatunku
    all_dfs = []
    for species in SPECIES:
        df_s = df.copy()
        df_s["species"] = species
        df_s["bite_probability"] = df_s.apply(
            lambda row: compute_bite_probability(row, species), axis=1
        )
        all_dfs.append(df_s)

    result = pd.concat(all_dfs, ignore_index=True)
    print(f"  Łącznie: {len(result)} rekordów ({len(df)} godzin × {len(SPECIES)} gatunków)")
    return result

# ─────────────────────────────────────────────
# TRENING
# ─────────────────────────────────────────────

def train(df: pd.DataFrame, species: str) -> lgb.Booster:
    df_s = df[df["species"] == species].copy()
    df_s = df_s.dropna(subset=FEATURE_COLS + ["bite_probability"])

    X = df_s[FEATURE_COLS]
    y = df_s["bite_probability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data  = lgb.Dataset(X_test,  label=y_test)

    params = {
        "objective":     "regression",
        "metric":        "rmse",
        "learning_rate": 0.05,
        "num_leaves":    31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":  5,
        "reg_alpha":     0.1,
        "reg_lambda":    0.1,
        "verbose":       -1,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  [{species}] RMSE: {rmse:.1f}% | próbki: {len(X_train)} train / {len(X_test)} test")

    return model

def train_all(years_back: int = 3):
    print(f"[{datetime.now():%H:%M}] Generuję dane treningowe ({years_back} lata)...")
    df = generate_training_data(years_back)

    models = {}
    print("\nTrenuję modele...")
    for species in SPECIES:
        model = train(df, species)
        path  = os.path.join(MODEL_DIR, f"model_{species}.lgb")
        model.save_model(path)
        models[species] = model

    # Zapisz metadane
    meta = {
        "trained_at": datetime.now().isoformat(),
        "species":    SPECIES,
        "features":   FEATURE_COLS,
        "records":    len(df),
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModele zapisane w: {MODEL_DIR}")
    return models

def fine_tune_all_species():
    print(f"[{datetime.now():%H:%M}] ROZPOCZYNAM GLOBALNY FINE-TUNING DLA RZEKI DEE...")
    
    # 1. Wczytanie danych z Australii
    df_au = load_au_data(FILE_NAME_AU)
    if df_au is None: return

    # 2. Pobranie pogody historycznej dla zakresu dat z pliku CSV
    start_str = df_au['date'].min().strftime('%Y-%m-%d')
    end_str = df_au['date'].max().strftime('%Y-%m-%d')
    print(f" Pobieram pogodę historyczną od {start_str} do {end_str}...")
    df_meteo = fetch_weather(start_str, end_str)
    
    # 3. Łączenie danych rzecznych z pogodowymi
    # Suffixes zapobiegną nadpisywaniu kolumn o tych samych nazwach
    df = pd.merge(df_au, df_meteo, left_on='date', right_on='timestamp', how='inner', suffixes=('', '_meteo'))
    
    # --- PRZYPISANIE SENSORÓW (TWOJE DANE) ---
    # To jest Twoja prawdziwa temperatura wody z rzeki
    df['water_temp_sensor'] = df['water_temp'] 
    
    # --- NAPRAWA POGODY (OPEN-METEO) ---
    # Jeśli meteo kłamie (mróz w Queensland), wyrównujemy temp. powietrza do wody
    mask_error = (df['temperature_c'] < 5) & (df['water_temp'] > 15)
    df.loc[mask_error, 'temperature_c'] = df['water_temp'] - 2
    
    # Wyliczamy cechy (księżyc, ciśnienie itp.)
    df = add_features(df)
    
    # --- KLUCZOWY KROK ---
    # Nadpisujemy wyliczoną przez funkcję 'water_temp_c' Twoim realnym odczytem z sensora.
    # Dzięki temu model będzie się uczył na PRAWDZIE, a nie na estymacji.
    df['water_temp_c'] = df['water_temp_sensor']
    
# --- PEŁNY PODGLĄD WSZYSTKICH DANYCH (DEBUG) ---
    print("\n" + "!"*80)
    print("KONTROLA WSZYSTKICH KOLUMN WEJŚCIOWYCH (Pierwsze 5 wierszy):")
    
    # Ustawienie wyświetlania tak, aby nie ucinało kolumn w konsoli
    pd.set_option('display.max_columns', None)  # Pokaż wszystkie kolumny
    pd.set_option('display.width', 1000)        # Szerokość linii w konsoli
    
    # Wyświetlamy 5 pierwszych wierszy - WSZYSTKIE KOLUMNY
    print(df.head(5))
    
    print("\n" + "-"*80)
    print("LISTA WSZYSTKICH WYKRYTYCH KOLUMN:")
    print(df.columns.tolist())
    
    # Sprawdzenie czy są jakieś puste wartości (NaN), które psują model
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n[!] OSTRZEŻENIE: Znaleziono brakujące dane (NaN):")
        print(missing[missing > 0])
    else:
        print("\n[OK] Brak pustych wartości (NaN) we wszystkich kolumnach.")
    
    print("!"*80 + "\n")
    # -----------------------------------------------
    
    # 4. Obliczanie tlenu i celu (target)
    df['est_do'] = 14.6 - 0.45 * df['water_temp_c'] + 0.008 * (df['water_temp_c']**2)
    df['target'] = (df['fish_count'] / 25 * 100).clip(upper=100)

    # 5. Pętla dotrenowania dla każdego gatunku
    for s in SPECIES:
        base_model_path = os.path.join(MODEL_DIR, f"model_{s}.lgb")
        if not os.path.exists(base_model_path):
            continue

        print(f" -> Adaptacja modelu: {s.upper()} do warunków Dee River...")
        
        # Tworzymy zbiór danych (teraz df ma już wszystkie FEATURE_COLS)
        train_data = lgb.Dataset(df[FEATURE_COLS], label=df['target'])

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.01,
            "verbose": -1
        }

        tuned_model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            init_model=base_model_path
        )

        save_path = os.path.join(MODEL_DIR, f"model_{s}_tuned.lgb")
        tuned_model.save_model(save_path)
        print(f"    Zapisano: {save_path}")

    print("\n[SUKCES] Modele zostały pomyślnie dostrojone!")
    
# ─────────────────────────────────────────────
# PREDYKCJA
# ─────────────────────────────────────────────

def load_models() -> dict:
    models = {}
    for species in SPECIES:
        # 1. Najpierw szukamy modelu dostrojonego do Twojej rzeki
        path_tuned = os.path.join(MODEL_DIR, f"model_{species}_tuned.lgb")
        # 2. Jeśli go nie ma, szukamy modelu bazowego (syntetycznego)
        path_base = os.path.join(MODEL_DIR, f"model_{species}.lgb")
        
        if os.path.exists(path_tuned):
            path = path_tuned
            print(f" [INFO] Ładuję model DOSTROJONY dla: {species}")
        elif os.path.exists(path_base):
            path = path_base
            print(f" [INFO] Ładuję model BAZOWY dla: {species}")
        else:
            raise FileNotFoundError(
                f"Brak modelu dla {species}. Uruchom: python fish_model.py --train-all"
            )
            
        m = lgb.Booster(model_file=path)
        models[species] = m
    return models

def predict_now(species_filter: str = None) -> pd.DataFrame:
    """Przewiduje prawdopodobieństwo brania dla kolejnych 24h."""
    print(f"[{datetime.now():%H:%M}] Pobieranie danych pogodowych...")
    df = fetch_recent_weather()
    df = add_features(df)

    # Ogranicz do następnych 24h od teraz
    now     = pd.Timestamp.now().floor("h")
    end_24h = now + pd.Timedelta(hours=24)
    df      = df[(df["timestamp"] >= now) & (df["timestamp"] < end_24h)]

    if len(df) == 0:
        raise ValueError("Brak danych dla przyszłych godzin")

    models   = load_models()
    species_list = [species_filter] if species_filter else SPECIES

    results = []
    for _, row in df.iterrows():
        entry = {
            "timestamp": str(row["timestamp"]),
            "hour":      row["timestamp"].hour,
        }
        for species in species_list:
            X    = pd.DataFrame([row[FEATURE_COLS]])
            prob = float(models[species].predict(X)[0])
            prob = round(np.clip(prob, 0, 100), 1)
            entry[species] = prob
        results.append(entry)

    return pd.DataFrame(results)

def print_forecast(df: pd.DataFrame):
    species_cols = [c for c in df.columns if c in SPECIES]
    header = f"{'Godzina':<20}" + "".join(f"{s.capitalize():>12}" for s in species_cols)
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    for _, row in df.iterrows():
        ts   = row["timestamp"][:16]
        vals = ""
        for s in species_cols:
            v = row[s]
            # Kolorowe wskaźniki
            if v >= 70:
                indicator = "🟢"
            elif v >= 40:
                indicator = "🟡"
            else:
                indicator = "🔴"
            vals += f"{indicator}{v:>5.0f}%    "
        print(f"  {ts}  {vals}")

    print(f"\n🟢 ≥70% doskonałe | 🟡 40-69% dobre | 🔴 <40% słabe")

    # Podsumowanie — najlepsza godzina dla każdego gatunku
    print(f"\n{'─'*40}")
    print("NAJLEPSZA GODZINA:")
    for s in species_cols:
        best_idx  = df[s].idxmax()
        best_row  = df.loc[best_idx]
        best_time = best_row["timestamp"][11:16]
        best_val  = best_row[s]
        print(f"  {s.capitalize():<10} {best_time}  →  {best_val:.0f}%")

def save_forecast_json(df: pd.DataFrame):
    path = os.path.join(MODEL_DIR, "fish_forecast.json")
    records = df.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "location":     {"lat": LAT, "lon": LON},
            "forecast":     records,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nJSON zapisany: {path}")




# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish bite predictor")
    parser.add_argument("--train",   action="store_true", help="Trenuj modele")
    parser.add_argument("--train-all",   action="store_true", help="Trenuj modele")
    parser.add_argument("--predict", action="store_true", help="Predykcja na teraz")
    parser.add_argument("--species", type=str, default=None,
                        help=f"Filtruj gatunek: {', '.join(SPECIES)}")
    parser.add_argument("--years",   type=int, default=3,
                        help="Ile lat danych treningowych (domyślnie 3)")
    args = parser.parse_args()

    if args.train:
        train_all(years_back=args.years)

    if args.predict:
        forecast_df = predict_now(species_filter=args.species)
        print_forecast(forecast_df)
        save_forecast_json(forecast_df)

    if not args.train and not args.predict and not args.train_all:
        parser.print_help()
    
    if args.train_all: # np. nowa flaga --train-all
        train_all()           # Najpierw biologia (syntetyczne)
        fine_tune_all_species() # Potem adaptacja do Australii