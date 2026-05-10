"""
patchtst_model.py — wspólna logika modelu PatchTST do prognozy 24h godzinowej.
Zamiennik weather_model.py / TCN.

Architektura: PatchTST (Huggingface transformers)
- Input:  168h (7 dni) historii, 22 cechy
- Patch:  24h (jeden dzień na patch) → 7 patchy
- Output: 24h prognozy × 9 zmiennych
"""
import os
import math
import numpy as np
import pandas as pd
import torch
import joblib
import requests
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from transformers import PatchTSTConfig, PatchTSTForPrediction

# =============================
# KONFIGURACJA
# =============================

LAT, LON      = 53.0138, 18.5981   # Toruń
INPUT_HOURS   = 168   # 7 dni historii
OUTPUT_HOURS  = 24    # prognoza 24h
PATCH_LENGTH  = 24    # 1 patch = 1 dzień
PATCH_STRIDE  = 24    # bez nakładania
BATCH_SIZE    = 8
EPOCHS        = 500
LR            = 0.00005

FEATURES_IN = [
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_gust", "rain_rate",
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
    "d_pressure_3h", "d_pressure_24h", "d_temp_6h",
    "wind_dir_sin", "wind_dir_cos",
    "temp_daily_max",
    "temp_daily_min",
    "cloudcover",
    "dewpoint",
    "apparent_temp",
    "solar_heating",
    "hour_shifted",
]

TARGETS = [
    "temperature",
    "wind_speed",
    "wind_gust",
    "pressure",
    "rain_rate",
    "humidity",
    "cloudcover",
    "apparent_temp",
    "dewpoint",
]

MODEL_DIR       = os.path.join(os.path.dirname(__file__), "patchtst_files")
MODEL_PATH      = os.path.join(MODEL_DIR, "patchtst_weather.pth")
SCALER_IN_PATH  = os.path.join(MODEL_DIR, "scaler_in.save")
SCALER_T_PATH   = os.path.join(MODEL_DIR, "scaler_t.save")
FORECAST_PATH   = os.path.join(MODEL_DIR, "latest_forecast.json")
CONFIG_PATH     = os.path.join(MODEL_DIR, "patchtst_config.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# FETCH DANYCH — identyczne jak w weather_model.py
# =============================

_RAW_COLS = ["temperature", "humidity", "pressure", "wind_speed", "wind_gust",
             "rain_rate", "wind_dir_sin", "wind_dir_cos",
             "cloudcover", "apparent_temp", "dewpoint", "shortwave_radiation"]
_UV_COLS  = ["uv_index"]


def _parse_response(h):
    wd10 = np.array(h["winddirection_10m"], dtype=float)
    return pd.DataFrame({
        "timestamp":           pd.to_datetime(h["time"]),
        "temperature":         h["temperature_2m"],
        "humidity":            h["relativehumidity_2m"],
        "pressure":            h["pressure_msl"],
        "wind_speed":          h["windspeed_10m"],
        "wind_gust":           h["windgusts_10m"],
        "rain_rate":           h["precipitation"],
        "wind_dir_sin":        np.sin(np.deg2rad(wd10)),
        "wind_dir_cos":        np.cos(np.deg2rad(wd10)),
        "cloudcover":          h["cloudcover"],
        "uv_index":            h["uv_index"],
        "apparent_temp":       h["apparent_temperature"],
        "dewpoint":            h["dewpoint_2m"],
        "shortwave_radiation": h.get("shortwave_radiation", [0.0] * len(h["time"])),
    })


def _fetch_with_retry(url: str, max_retries: int = 6) -> dict:
    import time
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=20)
        except requests.exceptions.ConnectionError:
            wait = 2 ** (attempt + 2)
            print(f"  Blad sieci - czekam {wait}s...")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = 2 ** (attempt + 3)
            print(f"  Rate limit (429) - czekam {wait}s...")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            wait = 2 ** (attempt + 1)
            print(f"  Blad serwera ({r.status_code}) - czekam {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise Exception(f"Nie udalo sie pobrac danych po {max_retries} probach")


def _fetch_archive(start_date, end_date):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relativehumidity_2m,pressure_msl,"
        f"windspeed_10m,windgusts_10m,precipitation,winddirection_10m,"
        f"cloudcover,uv_index,apparent_temperature,dewpoint_2m,shortwave_radiation"
        f"&start_date={start_date}&end_date={end_date}&timezone=UTC"
    )
    return _parse_response(_fetch_with_retry(url)["hourly"])


def _fetch_recent():
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relativehumidity_2m,pressure_msl,"
        f"windspeed_10m,windgusts_10m,precipitation,winddirection_10m,"
        f"cloudcover,uv_index,apparent_temperature,dewpoint_2m,shortwave_radiation"
        f"&past_days=7&forecast_days=1&timezone=UTC"
    )
    return _parse_response(_fetch_with_retry(url)["hourly"])


def _clean(df):
    df[_RAW_COLS] = (
        df[_RAW_COLS]
        .set_index(df["timestamp"])
        .interpolate(method="time")
        .ffill().bfill()
        .values
    )
    if "uv_index" in df.columns:
        df["uv_index"] = df["uv_index"].fillna(0.0)
    return df


def _fetch_from_influx(years_back: int = 5):
    try:
        from influxdb_client import InfluxDBClient
        client = InfluxDBClient(
            url   = "http://localhost:8086",
            token = "mH1sJUpajjdlKcQrM64YVu8efBZymCm--X0Jp2nRoaHFIZduitvapbZATXrA6t2TxnwQ2EVJ8RuxrfNM4efeDA==",
            org   = "zlote-branie",
        )
        flux = f"""
from(bucket: "historical_data")
  |> range(start: -{years_back}y)
  |> filter(fn: (r) => r["_measurement"] == "atmosphere")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""
        tables = client.query_api().query(flux)
        rows   = [row.values for table in tables for row in table.records]
        client.close()
        if len(rows) < 1000:
            return None
        df = pd.DataFrame(rows)
        df = df.rename(columns={
            "_time":          "timestamp",
            "temperature_c":  "temperature",
            "humidity_pct":   "humidity",
            "pressure_hpa":   "pressure",
            "wind_speed_kmh": "wind_speed",
            "wind_gust_kmh":  "wind_gust",
            "rain_rate_mm":   "rain_rate",
            "cloudcover_pct": "cloudcover",
            "shortwave_rad":  "shortwave_radiation",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"  InfluxDB: {len(df)} rekordow")
        return df
    except Exception as e:
        print(f"  InfluxDB niedostepny ({e}) - pobieram z API")
        return None


def _fix_influx_cols(df):
    if "wind_direction" in df.columns and "wind_dir_sin" not in df.columns:
        wd = df["wind_direction"].fillna(0)
        df["wind_dir_sin"] = np.sin(np.deg2rad(wd))
        df["wind_dir_cos"] = np.cos(np.deg2rad(wd))
    if "apparent_temp" not in df.columns:
        df["apparent_temp"] = df["temperature"]
    if "dewpoint" not in df.columns:
        T  = df["temperature"]
        RH = df["humidity"].clip(1, 100)
        df["dewpoint"] = T - ((100 - RH) / 5.0)
    return df


def fetch_training_data(years_back=5):
    import time
    today  = date.today()
    frames = []

    df_influx = _fetch_from_influx(years_back)
    if df_influx is not None and len(df_influx) > 8760:
        frames.append(df_influx)
        try:
            import signal
            def _timeout_handler(s, f): raise TimeoutError()
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)
            frames.append(_fetch_recent())
            signal.alarm(0)
            print(f"  + {len(frames[-1])}h z API")
        except Exception as e:
            try: signal.alarm(0)
            except: pass
            print(f"  Pomijam ostatnie 7 dni ({type(e).__name__})")
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = _fix_influx_cols(df)
        return _clean(df)

    print("  Pobieram z Open-Meteo API...")
    for year in range(today.year - years_back, today.year):
        print(f"  Rok {year}...", end=" ", flush=True)
        chunk = _fetch_archive(f"{year}-01-01", f"{year}-12-31")
        print(f"{len(chunk)}h")
        frames.append(chunk)
        time.sleep(3)
    archive_end = (today - timedelta(days=6)).strftime("%Y-%m-%d")
    frames.append(_fetch_archive(f"{today.year}-01-01", archive_end))
    frames.append(_fetch_recent())
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return _clean(df)


def fetch_recent_window():
    df = _fetch_recent()
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = _clean(df)
    now_loc = pd.Timestamp.now().floor("h")
    df = df[df["timestamp"] <= now_loc].reset_index(drop=True)
    return df


def add_features(df):
    h   = df["timestamp"].dt.hour
    doy = df["timestamp"].dt.day_of_year
    df["hour_sin"]       = np.sin(2 * np.pi * h   / 24)
    df["hour_cos"]       = np.cos(2 * np.pi * h   / 24)
    df["doy_sin"]        = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"]        = np.cos(2 * np.pi * doy / 365)
    df["d_pressure_3h"]  = df["pressure"].diff(3).fillna(0)
    df["d_pressure_24h"] = df["pressure"].diff(24).fillna(0)
    df["d_temp_6h"]      = df["temperature"].diff(6).fillna(0)
    df["temp_daily_max"] = df["temperature"].rolling(24, min_periods=1).max()
    df["temp_daily_min"] = df["temperature"].rolling(24, min_periods=1).min()
    if "shortwave_radiation" in df.columns:
        df["solar_heating"] = df["shortwave_radiation"].rolling(3, min_periods=1).mean().fillna(0)
    else:
        solar_proxy = (1 - df["cloudcover"] / 100) * np.maximum(0, -np.cos(2 * np.pi * h / 24))
        df["solar_heating"] = solar_proxy.rolling(3, min_periods=1).mean().fillna(0)
    h_shifted = (h + 2.5) % 24
    df["hour_shifted"] = np.sin(2 * np.pi * h_shifted / 24)
    return df


# =============================
# MODEL PATCHTST
# =============================

def build_config(n_features: int) -> PatchTSTConfig:
    num_patches = (INPUT_HOURS - PATCH_LENGTH) // PATCH_STRIDE + 1  # = 7 przy 168/24/24
    return PatchTSTConfig(
        num_input_channels   = n_features,
        context_length       = INPUT_HOURS,
        patch_length         = PATCH_LENGTH,
        patch_stride         = PATCH_STRIDE,
        prediction_length    = OUTPUT_HOURS,
        # Architektura — mała, pasuje do 2GB RAM
        d_model              = 32,
        num_attention_heads  = 4,
        num_hidden_layers    = 2,
        ffn_dim              = 64,
        dropout              = 0.3,
        head_dropout         = 0.3,
        # Multi-output: każdy kanał przewiduje siebie (channel-independence)
        # Dla TARGETS wykorzstujemy tylko pierwsze len(TARGETS) kanałów
        channel_attention    = False,
        scaling              = "std",   # normalizacja per-patch — kluczowa dla PatchTST
        loss                 = "mse",
        pre_norm             = True,
        norm_type            = "batchnorm",
    )


def build_model(n_features: int) -> PatchTSTForPrediction:
    config = build_config(n_features)
    model  = PatchTSTForPrediction(config)
    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Brak modelu PatchTST: {MODEL_PATH}\n"
            "Uruchom najpierw: python patchtst_train.py"
        )
    n_in   = len(FEATURES_IN)
    model  = build_model(n_in)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state  = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, device


# =============================
# PREDYKCJA
# =============================

def predict(df, df_input=None):
    """
    Przyjmuje DataFrame z >= INPUT_HOURS godzinami danych.
    Zwraca DataFrame z prognozą na 24h.
    """
    model, device = load_model()
    scaler_in     = joblib.load(SCALER_IN_PATH)
    scalers_out   = joblib.load(SCALER_T_PATH)

    # Okno wejściowe: ostatnie INPUT_HOURS godzin
    window_raw = df[FEATURES_IN].values[-INPUT_HOURS:]
    window     = scaler_in.transform(
        pd.DataFrame(window_raw, columns=FEATURES_IN)
    ).astype(np.float32)

    # PatchTST oczekuje: [batch, time, channels]
    X = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(past_values=X)
        # prediction_outputs: [batch, pred_len, num_channels]
        pred = out.prediction_outputs[0].cpu().numpy()   # [24, n_features]

    # Użyj tylko kanałów odpowiadających TARGETS
    # PatchTST w trybie channel-independence każdy kanał przewiduje siebie
    # TARGETS są podzbiorem FEATURES_IN — znajdź indeksy
    target_indices = [FEATURES_IN.index(t) for t in TARGETS]
    pred_targets   = pred[:, target_indices]   # [24, 9]

    # Timestamps
    now_h    = pd.Timestamp.now().floor("h")
    start_ts = now_h + pd.Timedelta(hours=1)
    ts       = pd.date_range(start_ts, periods=OUTPUT_HOURS, freq="h")

    # ── NWP z Open-Meteo jako blending reference ────────────────────────────
    # Pobierz prognozę NWP dla następnych 24h (forecast_days=2, przytnij do 24h)
    nwp_vals = {}
    try:
        nwp_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={LAT}&longitude={LON}"
            f"&hourly=temperature_2m,windspeed_10m,windgusts_10m,pressure_msl,"
            f"precipitation,relativehumidity_2m,cloudcover,apparent_temperature,dewpoint_2m"
            f"&forecast_days=2&timezone=UTC"
        )
        nwp_h   = _fetch_with_retry(nwp_url)["hourly"]
        nwp_df  = pd.DataFrame({
            "timestamp":    pd.to_datetime(nwp_h["time"]),
            "temperature":  nwp_h["temperature_2m"],
            "wind_speed":   nwp_h["windspeed_10m"],
            "wind_gust":    nwp_h["windgusts_10m"],
            "pressure":     nwp_h["pressure_msl"],
            "rain_rate":    nwp_h["precipitation"],
            "humidity":     nwp_h["relativehumidity_2m"],
            "cloudcover":   nwp_h["cloudcover"],
            "apparent_temp":nwp_h["apparent_temperature"],
            "dewpoint":     nwp_h["dewpoint_2m"],
        })
        nwp_df = nwp_df[nwp_df["timestamp"] > pd.Timestamp.utcnow().tz_localize(None).floor("h")]
        nwp_df = nwp_df.head(OUTPUT_HOURS).reset_index(drop=True)
        for t in TARGETS:
            if t in nwp_df.columns and len(nwp_df) >= OUTPUT_HOURS:
                nwp_vals[t] = nwp_df[t].values
    except Exception:
        pass  # brak NWP — użyj samego modelu

    # Wagi NWP: im wcześniej tym bardziej ufamy NWP (maleje z czasem)
    def nwp_weight(hour_i: int, target: str) -> float:
        if "temp" in target or target == "apparent_temp" or target == "dewpoint":
            base = 0.75
        elif target == "pressure":
            base = 0.70
        elif target in ("wind_speed", "wind_gust"):
            base = 0.65
        elif target == "rain_rate":
            base = 0.80
        else:
            base = 0.55
        # Liniowy zanik: h=0 → base, h=23 → base/2
        return base * (1 - 0.5 * hour_i / (OUTPUT_HOURS - 1))

    result = {"timestamp": ts}
    recent_obs = df[TARGETS].values[-6:]

    for i, name in enumerate(TARGETS):
        # Inverse transform przez skaler wejściowy (kolumna odpowiadająca TARGETS)
        feat_idx = FEATURES_IN.index(name)
        sc       = scalers_out[name]
        vals     = sc.inverse_transform(pred_targets[:, i].reshape(-1, 1)).flatten()

        # Bias correction
        obs_mean  = recent_obs[:, i].mean()
        pred_mean = vals[:6].mean()
        bias      = obs_mean - pred_mean
        vals      = vals + bias * 0.5

        # NWP blending
        if name in nwp_vals and len(nwp_vals[name]) == OUTPUT_HOURS:
            nwp = nwp_vals[name]
            blended = np.array([
                nwp_weight(h, name) * nwp[h] + (1 - nwp_weight(h, name)) * vals[h]
                for h in range(OUTPUT_HOURS)
            ])
            vals = blended

        # Fizyczne ograniczenia
        if name == "rain_rate":
            vals = np.where(vals < 0.1, 0.0, vals)
        if name in ("wind_speed", "wind_gust"):
            vals = np.maximum(0, vals)
        if name == "humidity":
            vals = np.clip(vals, 0, 100)
        if name == "cloudcover":
            vals = np.clip(vals, 0, 100)
        if name == "apparent_temp":
            vals = np.clip(vals, -50, 60)

        result[name] = vals

    if "dewpoint" in result and "temperature" in result:
        result["dewpoint"] = np.minimum(result["dewpoint"], result["temperature"] - 0.1)

    # UV analityczny
    cc_vals = result.get("cloudcover", np.full(OUTPUT_HOURS, 50.0))
    uv_vals = []
    lat_r   = math.radians(LAT)
    for i, t in enumerate(pd.DatetimeIndex(ts)):
        doy        = t.day_of_year
        hour_utc   = t.hour + t.minute / 60
        decl       = math.radians(23.45 * math.sin(math.radians(360 / 365 * (doy - 81))))
        hour_angle = math.radians(15 * (hour_utc - 12))
        sin_elev   = max(0, (math.sin(lat_r) * math.sin(decl) +
                             math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle)))
        uv_max       = max(0, 3.0 + 4.0 * math.sin(math.radians(360 / 365 * (doy - 80))))
        cloud_factor = 1.0 - 0.75 * ((cc_vals[i] / 100.0) ** 3.4)
        uv_vals.append(max(0.0, round(uv_max * sin_elev * cloud_factor, 1)))
    result["uv_index"] = np.array(uv_vals)

    # PoP
    rain_arr = result["rain_rate"]
    hum_arr  = result["humidity"]
    cc_arr   = result["cloudcover"]
    pop_arr  = np.zeros(OUTPUT_HOURS)
    for i in range(OUTPUT_HOURS):
        pop = 0.0
        if rain_arr[i] > 0.5:
            pop = min(95, 40 + rain_arr[i] * 10)
        elif rain_arr[i] > 0.1:
            pop = 20
        pop += max(0, (hum_arr[i] - 70) * 0.5)
        pop += max(0, (cc_arr[i]  - 60) * 0.3)
        pop_arr[i] = min(99, max(0, pop))
    result["pop_pct"] = pop_arr

    return pd.DataFrame(result)