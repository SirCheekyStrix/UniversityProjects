import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tcn import TCN
import joblib
import requests

# =============================
# KONFIGURACJA
# =============================

INPUT_HOURS  = 168   # 7 dni kontekstu
OUTPUT_HOURS = 24    # prognoza 24h
BATCH_SIZE   = 64
EPOCHS       = 2000
LR           = 0.0001

FEATURES_IN = [
    # Podstawowe
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_gust", "rain_rate",
    # Czas
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
    # Proxy frontów — tylko pochodne ciśnienia i temperatury
    # (bezpłatne, nie wymagają extra API, sprawdzone fizycznie)
    "d_pressure_3h",
    "d_pressure_24h",
    "d_temp_6h",
    # Kierunek wiatru (masa powietrza)
    "wind_dir_sin",
    "wind_dir_cos",
    # Temperatura 850hPa — najsilniejszy proxy adwekcji ciepła
    "temp_850hpa",
]

TARGET         = "temperature"
MODEL_PATH     = "tcn_weather.pth"
SCALER_IN_PATH = "scaler_in.save"
SCALER_T_PATH  = "scaler_t.save"

# =============================
# FETCH DANYCH
# =============================

def fetch_open_meteo(lat, lon, days_back=None):
    """
    Pobiera dane historyczne z Open-Meteo Archive API (lata wstecz)
    oraz aktualne dane z Forecast API i skleja je w jeden DataFrame.
    """
    from datetime import date, timedelta

    num_cols = ["temperature", "humidity", "pressure", "wind_speed", "wind_gust",
                "rain_rate", "wind_dir_sin", "wind_dir_cos",
                "temp_850hpa", "u_wind_850hpa", "v_wind_850hpa", "geopotential_500"]

    def clean(df):
        df[num_cols] = (
            df[num_cols]
            .set_index(df["timestamp"])
            .interpolate(method="time")
            .ffill().bfill()
            .values
        )
        return df

    def fetch_chunk(start_date, end_date):
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relativehumidity_2m,pressure_msl,"
            f"windspeed_10m,windgusts_10m,precipitation,winddirection_10m,"
            f"temperature_850hPa,windspeed_850hPa,winddirection_850hPa,"
            f"geopotential_height_500hPa"
            f"&start_date={start_date}&end_date={end_date}&timezone=UTC"
        )
        r = requests.get(url)
        r.raise_for_status()
        d = r.json()
        h = d["hourly"]
        # Składowe U/V wiatru 850hPa z prędkości i kierunku
        ws850  = np.array(h["windspeed_850hPa"],    dtype=float)
        wd850  = np.array(h["winddirection_850hPa"], dtype=float)
        u850   = -ws850 * np.sin(np.deg2rad(wd850))
        v850   = -ws850 * np.cos(np.deg2rad(wd850))
        wd10   = np.array(h["winddirection_10m"], dtype=float)
        return pd.DataFrame({
            "timestamp":        pd.to_datetime(h["time"]),
            "temperature":      h["temperature_2m"],
            "humidity":         h["relativehumidity_2m"],
            "pressure":         h["pressure_msl"],
            "wind_speed":       h["windspeed_10m"],
            "wind_gust":        h["windgusts_10m"],
            "rain_rate":        h["precipitation"],
            "wind_dir_sin":     np.sin(np.deg2rad(wd10)),
            "wind_dir_cos":     np.cos(np.deg2rad(wd10)),
            "temp_850hpa":      h["temperature_850hPa"],
            "u_wind_850hpa":    u850,
            "v_wind_850hpa":    v850,
            "geopotential_500": h["geopotential_height_500hPa"],
        })

    def fetch_forecast():
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relativehumidity_2m,pressure_msl,"
            f"windspeed_10m,windgusts_10m,precipitation,winddirection_10m,"
            f"temperature_850hPa,windspeed_850hPa,winddirection_850hPa,"
            f"geopotential_height_500hPa"
            f"&past_days=7&forecast_days=1&timezone=UTC"
        )
        r = requests.get(url)
        r.raise_for_status()
        d = r.json()
        h = d["hourly"]
        ws850  = np.array(h["windspeed_850hPa"],    dtype=float)
        wd850  = np.array(h["winddirection_850hPa"], dtype=float)
        u850   = -ws850 * np.sin(np.deg2rad(wd850))
        v850   = -ws850 * np.cos(np.deg2rad(wd850))
        wd10   = np.array(h["winddirection_10m"], dtype=float)
        return pd.DataFrame({
            "timestamp":        pd.to_datetime(h["time"]),
            "temperature":      h["temperature_2m"],
            "humidity":         h["relativehumidity_2m"],
            "pressure":         h["pressure_msl"],
            "wind_speed":       h["windspeed_10m"],
            "wind_gust":        h["windgusts_10m"],
            "rain_rate":        h["precipitation"],
            "wind_dir_sin":     np.sin(np.deg2rad(wd10)),
            "wind_dir_cos":     np.cos(np.deg2rad(wd10)),
            "temp_850hpa":      h["temperature_850hPa"],
            "u_wind_850hpa":    u850,
            "v_wind_850hpa":    v850,
            "geopotential_500": h["geopotential_height_500hPa"],
        })

    # Pobierz 5 lat historii z Archive API (rok po roku żeby nie przekroczyć limitu)
    today      = date.today()
    start_year = today.year - 5
    frames     = []

    for year in range(start_year, today.year):
        start = f"{year}-01-01"
        end   = f"{year}-12-31"
        print(f"  Pobieram rok {year}...", end=" ", flush=True)
        chunk = fetch_chunk(start, end)
        print(f"{len(chunk)} godzin")
        frames.append(chunk)

    # Bieżący rok z Archive (do ~5 dni temu) + Forecast (ostatnie 7 dni + jutro)
    archive_end = (today - timedelta(days=6)).strftime("%Y-%m-%d")
    print(f"  Pobieram {today.year} (do {archive_end})...", end=" ", flush=True)
    chunk = fetch_chunk(f"{today.year}-01-01", archive_end)
    print(f"{len(chunk)} godzin")
    frames.append(chunk)

    print(f"  Pobieram ostatnie 7 dni + prognoza jutro...", end=" ", flush=True)
    forecast_chunk = fetch_forecast()
    print(f"{len(forecast_chunk)} godzin")
    frames.append(forecast_chunk)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    nan_before = df[num_cols].isna().sum().sum()
    df = clean(df)
    nan_after  = df[num_cols].isna().sum().sum()
    print(f"  NaN: {nan_before} → {nan_after} (po interpolacji)")
    return df


def add_time_features(df):
    h   = df["timestamp"].dt.hour
    doy = df["timestamp"].dt.day_of_year
    df["hour_sin"] = np.sin(2 * np.pi * h   / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h   / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * doy / 365)

    # Pochodne ciśnienia — proxy nadchodzących frontów
    df["d_pressure_3h"]  = df["pressure"].diff(3).fillna(0)
    df["d_pressure_24h"] = df["pressure"].diff(24).fillna(0)

    # Zmiana temperatury — adwekcja masy powietrza
    df["d_temp_6h"] = df["temperature"].diff(6).fillna(0)

    return df

# =============================
# MODEL
# =============================

class WeatherTCN(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        # Większa sieć uzasadniona ~43 000 próbkami treningowymi
        channels = [64, 128, 128, 64]

        self.tcn = TCN(
            num_inputs=n_in,
            num_channels=channels,
            kernel_size=5,
            dropout=0.25,
        )

        with torch.no_grad():
            dummy   = torch.zeros(1, n_in, INPUT_HOURS)
            last_ch = self.tcn(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(last_ch, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.head(self.tcn(x)[:, :, -1])

# =============================
# SEKWENCJE
# =============================

def create_sequences(X_data, y_data):
    Xs, ys = [], []
    for i in range(len(X_data) - INPUT_HOURS - OUTPUT_HOURS + 1):
        Xs.append(X_data[i : i + INPUT_HOURS])
        ys.append(y_data[i + INPUT_HOURS : i + INPUT_HOURS + OUTPUT_HOURS])
    Xs = np.transpose(np.array(Xs), (0, 2, 1))  # [N, n_in, T]
    ys = np.array(ys)                             # [N, OUTPUT_HOURS]
    return Xs, ys

# =============================
# TRENING
# =============================

def train_model(df):
    scaler_in = MinMaxScaler()
    X_scaled  = scaler_in.fit_transform(df[FEATURES_IN])
    joblib.dump(scaler_in, SCALER_IN_PATH)

    scaler_t = MinMaxScaler()
    T_scaled = scaler_t.fit_transform(df[[TARGET]]).flatten()
    joblib.dump(scaler_t, SCALER_T_PATH)

    X, y = create_sequences(X_scaled, T_scaled)

    split   = int(len(X) * 0.85)   # więcej danych na trening
    X_tr, X_v = X[:split], X[split:]
    y_tr, y_v = y[:split], y[split:]

    to_t  = lambda a: torch.tensor(a, dtype=torch.float32)
    tr_dl = DataLoader(TensorDataset(to_t(X_tr), to_t(y_tr)), BATCH_SIZE, shuffle=True)
    v_dl  = DataLoader(TensorDataset(to_t(X_v),  to_t(y_v)),  BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = WeatherTCN(len(FEATURES_IN), OUTPUT_HOURS).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit   = nn.HuberLoss(delta=0.1)   # Odporny na outliery, lepszy niż MSE

    # Cosine annealing: LR opada płynnie przez cały trening
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=200, T_mult=2, eta_min=1e-6
    )

    best, patience_cnt, PATIENCE = float("inf"), 0, 100

    print(f"Trening na {device} | train={len(X_tr)} | val={len(X_v)} | features={len(FEATURES_IN)}")

    for ep in range(EPOCHS):
        model.train()
        tl = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        sched.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in v_dl:
                vl += crit(model(xb.to(device)), yb.to(device)).item()

        avg_t, avg_v = tl / len(tr_dl), vl / len(v_dl)

        if np.isnan(avg_t) or np.isnan(avg_v):
            print(f"NaN w epoce {ep+1}!")
            break

        if (ep + 1) % 50 == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"Ep {ep+1:04d} | Train {avg_t:.5f} | Val {avg_v:.5f} | LR {lr_now:.7f}")

        if avg_v < best:
            best = avg_v
            torch.save(model.state_dict(), MODEL_PATH)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stop @ ep {ep+1} | best val={best:.5f}")
                break

    import os
    if os.path.exists(MODEL_PATH) and best < float("inf"):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"\nNajlepszy val loss: {best:.5f}")

    # Szacowany błąd w °C
    t_range = scaler_t.data_range_[0]
    rmse_c  = np.sqrt(best) * t_range
    print(f"Szacowany RMSE temperatury: ~{rmse_c:.2f}°C")
    return model

# =============================
# PREDYKCJA
# =============================

def predict_future(model, df):
    scaler_in = joblib.load(SCALER_IN_PATH)
    scaler_t  = joblib.load(SCALER_T_PATH)

    window = scaler_in.transform(
        pd.DataFrame(df[FEATURES_IN].values[-INPUT_HOURS:], columns=FEATURES_IN)
    )
    X = torch.tensor(window.T[np.newaxis], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        pred = model(X.to(device)).cpu().numpy().reshape(-1, 1)

    temps = scaler_t.inverse_transform(pred).flatten()
    ts    = pd.date_range(
        df["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
        periods=OUTPUT_HOURS, freq="h"
    )
    return pd.DataFrame({"timestamp": ts, "temperature": temps})

# =============================
# WYKRES
# =============================

def plot_forecast(df, forecast_df):
    history_hours = 48
    hist_ts   = df["timestamp"].iloc[-history_hours:].values
    hist_temp = df["temperature"].values[-history_hours:]
    pred_ts   = forecast_df["timestamp"].values
    pred_temp = forecast_df["temperature"].values

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist_ts, hist_temp, color="steelblue", linewidth=2, label="Historia (48h)")
    ax.plot(
        np.concatenate([[hist_ts[-1]], pred_ts]),
        np.concatenate([[hist_temp[-1]], pred_temp]),
        color="crimson", linewidth=1.8, linestyle="--",
        marker="o", markersize=4, label="Prognoza AI (24h)"
    )
    ax.axvline(x=pd.Timestamp(hist_ts[-1]), color="black", linestyle=":", linewidth=1.2)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    ax.set_title("Prognoza temperatury — TCN")
    ax.set_ylabel("°C")
    ax.set_xlabel("Data i godzina (UTC)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================
# MAIN
# =============================

def main():
    import os
    for f in [MODEL_PATH, SCALER_IN_PATH, SCALER_T_PATH]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Usunięto stary plik: {f}")
    print("Pobieram dane (92 dni)...")
    df = fetch_open_meteo(lat=53.0138, lon=18.5981)
    df = add_time_features(df)
    print(f"Zbiór: {len(df)} godzin | cechy: {FEATURES_IN}")

    model       = train_model(df)
    forecast_df = predict_future(model, df)

    print("\nPrognoza temperatury 24h:")
    print(forecast_df.to_string(index=False))

    plot_forecast(df, forecast_df)


if __name__ == "__main__":
    main()