"""
weather_model.py — wspólna logika modelu, fetchowania i cech.
Importowany przez train.py i predict.py.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tcn import TCN
import joblib
import requests
from datetime import date, timedelta

# =============================
# KONFIGURACJA
# =============================

LAT, LON     = 53.0138, 18.5981   # Toruń — zmień na swoje koordynaty
INPUT_HOURS  = 72    # 3 doby historii (zamiast 7) → 4x mniej RAM
OUTPUT_HOURS = 24
BATCH_SIZE   = 16   # mały batch → mało RAM, działa na 2GB
EPOCHS       = 2000
LR           = 0.0001

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
TARGET  = "temperature"   # główny cel (temperatura)
TARGETS = [                # wszystkie przewidywane zmienne
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

# Ścieżki — wszystkie pliki modelu w jednym katalogu
MODEL_DIR      = os.path.join(os.path.dirname(__file__), "model_files")
MODEL_PATH     = os.path.join(MODEL_DIR, "tcn_weather.pth")
SCALER_IN_PATH = os.path.join(MODEL_DIR, "scaler_in.save")
SCALER_T_PATH  = os.path.join(MODEL_DIR, "scaler_t.save")   # dict skalerów dla TARGETS
FORECAST_PATH  = os.path.join(MODEL_DIR, "latest_forecast.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# FETCH DANYCH
# =============================

_RAW_COLS = ["temperature", "humidity", "pressure", "wind_speed", "wind_gust",
             "rain_rate", "wind_dir_sin", "wind_dir_cos",
             "cloudcover", "apparent_temp", "dewpoint", "shortwave_radiation"]
# uv_index osobno — NaN w nocy to wartość fizyczna (brak UV), nie brakujące dane
_UV_COLS = ["uv_index"]

def _parse_response(h):
    """Parsuje odpowiedź hourly z obu API do DataFrame."""
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
    """GET z exponential backoff przy 429/5xx."""
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
            print(f"  Rate limit (429) - czekam {wait}s (proba {attempt+1}/{max_retries})...")
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
    """Ostatnie 7 dni + prognoza jutro z Forecast API."""
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
    """Interpoluje NaN-y po czasie."""
    df[_RAW_COLS] = (
        df[_RAW_COLS]
        .set_index(df["timestamp"])
        .interpolate(method="time")
        .ffill().bfill()
        .values
    )
    # uv_index: NaN w nocy = 0 (brak promieniowania UV), nie interpolować
    if "uv_index" in df.columns:
        df["uv_index"] = df["uv_index"].fillna(0.0)
    return df

def _fetch_from_influx(years_back: int = 5):
    """Pobierz dane z InfluxDB historical_data — szybsze, bez rate limit."""
    try:
        import time as _time
        from influxdb_client import InfluxDBClient
        client = InfluxDBClient(
            url   = "http://localhost:8086",
            token = "mH1sJUpajjdlKcQrM64YVu8efBZymCm--X0Jp2nRoaHFIZduitvapbZATXrA6t2TxnwQ2EVJ8RuxrfNM4efeDA==",
            org   = "zlote-branie",
        )
        query_api = client.query_api()
        flux = f"""
from(bucket: "historical_data")
  |> range(start: -{years_back}y)
  |> filter(fn: (r) => r["_measurement"] == "atmosphere")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""
        tables = query_api.query(flux)
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


def fetch_training_data(years_back=5):
    """Pobiera dane treningowe — najpierw z InfluxDB, fallback do Open-Meteo API."""
    import time
    today  = date.today()
    frames = []

    # Priorytet: InfluxDB (szybkie, bez rate limit)
    df_influx = _fetch_from_influx(years_back)
    if df_influx is not None and len(df_influx) > 8760:
        frames.append(df_influx)
        # Doczytaj ostatnie dni z API — opcjonalne, pomin jesli wolne
        print(f"  Ostatnie 7 dni z API...", end=" ", flush=True)
        try:
            import signal
            def _timeout_handler(s, f): raise TimeoutError()
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)   # max 30 sekund na API call
            frames.append(_fetch_recent())
            signal.alarm(0)
            print(f"{len(frames[-1])}h")
        except Exception as e:
            try: signal.alarm(0)
            except: pass
            print(f"pominięto ({type(e).__name__}) — trenuje bez ostatnich 7 dni")
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return _clean(df)

    # Fallback: Open-Meteo API z opoznieniami miedzy latami
    print("  Pobieram z Open-Meteo API (moze zajac kilka minut)...")
    for year in range(today.year - years_back, today.year):
        print(f"  Rok {year}...", end=" ", flush=True)
        chunk = _fetch_archive(f"{year}-01-01", f"{year}-12-31")
        print(f"{len(chunk)}h")
        frames.append(chunk)
        time.sleep(3)   # unikaj rate limit

    archive_end = (today - timedelta(days=6)).strftime("%Y-%m-%d")
    print(f"  {today.year} do {archive_end}...", end=" ", flush=True)
    frames.append(_fetch_archive(f"{today.year}-01-01", archive_end))
    print(f"{len(frames[-1])}h")

    print(f"  Ostatnie 7 dni + jutro...", end=" ", flush=True)
    frames.append(_fetch_recent())
    print(f"{len(frames[-1])}h")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    nan_n = df[_RAW_COLS].isna().sum().sum()
    df    = _clean(df)
    print(f"  NaN: {nan_n} → {df[_RAW_COLS].isna().sum().sum()} (po interpolacji)")
    return df

def fetch_recent_window():
    """
    Pobiera tylko ostatnie 7 dni + jutro — używane przy predykcji co godzinę.
    Przycina dane do aktualnej godziny UTC żeby model wiedział która jest godzina.
    """
    df = _fetch_recent()
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = _clean(df)

    # Przytnij do aktualnej godziny UTC — model musi "widzieć" aktualną godzinę
    # jako ostatni rekord, nie dane z przyszłości (forecast_days zwraca jutro)
    now_utc = pd.Timestamp.utcnow().floor("h").tz_localize(None)
    df = df[df["timestamp"] <= now_utc].reset_index(drop=True)

    return df

def add_features(df):
    """Dodaje cechy czasowe i pochodne — działa na dowolnym DataFrame."""
    h   = df["timestamp"].dt.hour
    doy = df["timestamp"].dt.day_of_year
    df["hour_sin"]      = np.sin(2 * np.pi * h   / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * h   / 24)
    df["doy_sin"]       = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"]       = np.cos(2 * np.pi * doy / 365)
    df["d_pressure_3h"] = df["pressure"].diff(3).fillna(0)
    df["d_pressure_24h"]= df["pressure"].diff(24).fillna(0)
    df["d_temp_6h"]     = df["temperature"].diff(6).fillna(0)
    # ── Cechy fizyczne ──────────────────────────────────────────
    # Dobowe ekstremum z kroczącego okna 24h
    df["temp_daily_max"] = df["temperature"].rolling(24, min_periods=1).max()
    df["temp_daily_min"] = df["temperature"].rolling(24, min_periods=1).min()

    # Opóźnienie termiczne — szczyt temperatury jest ~2-3h po szczycie nasłonecznienia
    # solar_heating: skumulowane promieniowanie z ostatnich 3h (proxy cieplna bezwładność)
    if "shortwave_radiation" in df.columns:
        df["solar_heating"] = df["shortwave_radiation"].rolling(3, min_periods=1).mean().fillna(0)
    elif "cloudcover" in df.columns:
        # Jeśli nie ma radiation — przybliż z zachmurzenia i pory dnia
        # Niskie zachmurzenie + dzień = duże promieniowanie
        solar_proxy = (1 - df["cloudcover"] / 100) * np.maximum(0, -np.cos(2 * np.pi * h / 24))
        df["solar_heating"] = solar_proxy.rolling(3, min_periods=1).mean().fillna(0)
    else:
        df["solar_heating"] = 0.0

    # hour_shifted: sinus godziny przesuniętej o +2.5h (15h staje się nowym "południem")
    # Model widzi szczyt promieniowania o 12h, ale szczyt temp jest o 14-15h
    h_shifted = (h + 2.5) % 24
    df["hour_shifted"] = np.sin(2 * np.pi * h_shifted / 24)

    return df

# =============================
# MODEL
# =============================

class WeatherTCN(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        # Mały model dla 2GB RAM / 2 rdzeni
        channels = [32, 64, 32]
        self.tcn = TCN(num_inputs=n_in, num_channels=channels,
                       kernel_size=3, dropout=0.2)
        with torch.no_grad():
            last_ch = self.tcn(torch.zeros(1, n_in, INPUT_HOURS)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(last_ch, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32),     nn.GELU(),
            nn.Linear(32, n_out),
        )

    def forward(self, x):
        return self.head(self.tcn(x)[:, :, -1])

def load_model():
    """Wczytuje wytrenowany model z dysku. Rzuca błąd jeśli nie istnieje."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Brak modelu: {MODEL_PATH}\n"
            "Uruchom najpierw: python train.py"
        )
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_out   = OUTPUT_HOURS * len(TARGETS)
    model   = WeatherTCN(len(FEATURES_IN), n_out).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        raise RuntimeError(
            f"Model na dysku jest niezgodny z obecną architekturą (wyjść: {n_out}).\n"
            f"Uruchom: python train.py\n"
            f"Szczegóły: {e}"
        ) from None
    model.eval()
    return model, device

# =============================
# SEKWENCJE (tylko dla treningu)
# =============================

def create_sequences(X_data, y_data):
    Xs, ys = [], []
    for i in range(len(X_data) - INPUT_HOURS - OUTPUT_HOURS + 1):
        Xs.append(X_data[i : i + INPUT_HOURS])
        ys.append(y_data[i + INPUT_HOURS : i + INPUT_HOURS + OUTPUT_HOURS])
    return np.transpose(np.array(Xs), (0, 2, 1)), np.array(ys)

# =============================
# PREDYKCJA (używana przez predict.py)
# =============================

def predict(df):
    """
    Przyjmuje DataFrame z ostatnimi >= INPUT_HOURS godzinami danych,
    zwraca DataFrame z prognozą wszystkich zmiennych na 24h.
    """
    model, device = load_model()
    scaler_in      = joblib.load(SCALER_IN_PATH)
    scalers_out    = joblib.load(SCALER_T_PATH)   # dict {nazwa: scaler}

    window = scaler_in.transform(
        pd.DataFrame(df[FEATURES_IN].values[-INPUT_HOURS:], columns=FEATURES_IN)
    )
    X = torch.tensor(window.T[np.newaxis], dtype=torch.float32).to(device)

    with torch.no_grad():
        # pred: [1, OUTPUT_HOURS * len(TARGETS)]
        pred = model(X).cpu().numpy().reshape(OUTPUT_HOURS, len(TARGETS))

    # Prognoza od teraz+1h — ignoruj przyszłe godziny które API zwraca w forecast_days
    now_h    = pd.Timestamp.now().floor("h")
    start_ts = now_h + pd.Timedelta(hours=1)
    ts = pd.date_range(start_ts, periods=OUTPUT_HOURS, freq="h")
    result = {"timestamp": ts}

    # Kalibracja bias — porównaj ostatnie 6h predykcji z obserwacjami
    # Jeśli model systematycznie zawyża/zaniża, korygujemy offset
    recent_obs  = df[TARGETS].values[-6:]   # ostatnie 6h rzeczywistych danych

    for i, name in enumerate(TARGETS):
        vals = scalers_out[name].inverse_transform(pred[:, i].reshape(-1, 1)).flatten()

        # Bias correction: średnia różnica (obs - pred) z ostatnich 6h
        # Używamy predykcji modelu dla tych samych 6h jako referencji
        obs_mean  = recent_obs[:, i].mean()
        pred_mean = vals[:6].mean()
        bias      = obs_mean - pred_mean
        # Stosujemy tylko 50% korekty żeby nie przesadzić
        vals      = vals + bias * 0.5

        # Zabezpieczenia fizyczne
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

    # Punkt rosy zawsze <= temperatura (prawo fizyczne)
    if "dewpoint" in result and "temperature" in result:
        result["dewpoint"] = np.minimum(result["dewpoint"], result["temperature"] - 0.1)

    # UV index — oblicz analitycznie (model słabo uczy się UV bo 50% danych = noc = 0)
    import math
    timestamps = pd.DatetimeIndex(ts)
    cc_vals    = result.get("cloudcover", np.full(OUTPUT_HOURS, 50.0))
    uv_vals    = []
    lat_r      = math.radians(LAT)
    for i, t in enumerate(timestamps):
        doy        = t.day_of_year
        hour_utc   = t.hour + t.minute / 60
        decl       = math.radians(23.45 * math.sin(math.radians(360 / 365 * (doy - 81))))
        hour_angle = math.radians(15 * (hour_utc - 12))
        sin_elev   = max(0, (math.sin(lat_r) * math.sin(decl) +
                             math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle)))
        # Maks UV dla pory roku: ~2 w zimie, ~7 latem dla Torunia (52°N)
        uv_max       = max(0, 3.0 + 4.0 * math.sin(math.radians(360 / 365 * (doy - 80))))
        cloud_factor = 1.0 - 0.75 * ((cc_vals[i] / 100.0) ** 3.4)
        uv_vals.append(max(0.0, round(uv_max * sin_elev * cloud_factor, 1)))
    result["uv_index"] = np.array(uv_vals)

    return pd.DataFrame(result)