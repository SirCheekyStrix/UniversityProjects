"""
openmeteo_to_influx.py — pobiera dane z Open-Meteo symulujące wszystkie czujniki
i zapisuje do InfluxDB. Działa jako backfill historyczny i bieżące odświeżanie.

Dane symulowane (dopóki fizyczne czujniki nie istnieją):
    BME280          → temperature_c, humidity_pct, pressure_hpa
    BH1750/LTR390   → lux, uv_index, solar_radiation
    Tipping Bucket  → rain_rate_mm
    Wind sensor     → wind_speed_kmh, wind_gust_kmh, wind_direction
    DS18B20         → water_temp_c  (estymowana z temp powietrza)
    Depth sensor    → water_level_cm (estymowana z opadów)
    pH meter        → water_ph      (syntetyczna, zależna od opadów)
    TDS sensor      → water_tds     (estymowana z konduktywności)
    DO sensor       → water_oxygen  (estymowana z temp wody)
    Turbidity       → turbidity_ntu (estymowana z opadów)

Użycie:
    pip install influxdb-client requests pandas

    # Backfill 5 lat historii
    python openmeteo_to_influx.py --backfill --years 5

    # Ostatnie 7 dni
    python openmeteo_to_influx.py --backfill --days 7

    # Tylko bieżące dane (ostatnie 2 dni + prognoza)
    python openmeteo_to_influx.py --recent

    # Daemon — odświeżaj co godzinę
    python openmeteo_to_influx.py --daemon
"""

import argparse
import math
import time
import warnings
from datetime import date, datetime, timedelta, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# ─────────────────────────────────────────────
# KONFIGURACJA — zmień na swoje
# ─────────────────────────────────────────────
INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "mH1sJUpajjdlKcQrM64YVu8efBZymCm--X0Jp2nRoaHFIZduitvapbZATXrA6t2TxnwQ2EVJ8RuxrfNM4efeDA=="
INFLUX_ORG    = "zlote-branie"
INFLUX_BUCKET = "historical_data"

LAT, LON      = 53.0138, 18.5981      # Toruń
LOCATION      = "torun"

# Measurement names w InfluxDB
MEAS_ATMOSPHERE = "atmosphere"        # BME280, wiatr, deszcz, UV
MEAS_WATER      = "water"             # DS18B20, pH, TDS, DO, turbidity, poziom
MEAS_DERIVED    = "derived"           # cechy pochodne dla modeli

# Chunk API — ile dni pobieramy naraz (Archive API limit)
CHUNK_DAYS    = 365

# ─────────────────────────────────────────────
# POBIERANIE DANYCH Z OPEN-METEO
# ─────────────────────────────────────────────

def fetch_archive_chunk(start: str, end: str) -> pd.DataFrame:
    """Pobiera dane godzinowe z Archive API."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relativehumidity_2m,pressure_msl,"
        f"windspeed_10m,windgusts_10m,winddirection_10m,"
        f"precipitation,cloudcover,shortwave_radiation,direct_radiation,"
        f"apparent_temperature,dewpoint_2m,"
        f"temperature_850hPa,windspeed_850hPa,winddirection_850hPa,"
        f"geopotential_height_500hPa"
        f"&start_date={start}&end_date={end}&timezone=UTC"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({
        "timestamp":         pd.to_datetime(h["time"]).tz_localize("UTC"),
        "temperature_c":     pd.array(h["temperature_2m"],      dtype=float),
        "humidity_pct":      pd.array(h["relativehumidity_2m"],  dtype=float),
        "pressure_hpa":      pd.array(h["pressure_msl"],         dtype=float),
        "wind_speed_kmh":    pd.array(h["windspeed_10m"],        dtype=float),
        "wind_gust_kmh":     pd.array(h["windgusts_10m"],        dtype=float),
        "wind_direction":    pd.array(h["winddirection_10m"],    dtype=float),
        "rain_rate_mm":      pd.array(h["precipitation"],        dtype=float),
        "cloudcover_pct":    pd.array(h["cloudcover"],           dtype=float),
        "shortwave_rad":     pd.array(h["shortwave_radiation"],  dtype=float),
        "direct_rad":        pd.array(h["direct_radiation"],     dtype=float),
        "apparent_temp":     pd.array(h["apparent_temperature"], dtype=float),
        "dewpoint":          pd.array(h["dewpoint_2m"],          dtype=float),
        "temp_850hpa":       pd.array(h["temperature_850hPa"],   dtype=float),
        "wind_dir_sin":      np.sin(np.deg2rad(pd.array(h["winddirection_10m"], dtype=float))),
        "wind_dir_cos":      np.cos(np.deg2rad(pd.array(h["winddirection_10m"], dtype=float))),
        "u_wind_850hpa":    -pd.array(h["windspeed_850hPa"], dtype=float) * np.sin(np.deg2rad(pd.array(h["winddirection_850hPa"], dtype=float))),
        "v_wind_850hpa":    -pd.array(h["windspeed_850hPa"], dtype=float) * np.cos(np.deg2rad(pd.array(h["winddirection_850hPa"], dtype=float))),
        "geopotential_500":  pd.array(h["geopotential_height_500hPa"], dtype=float),
    })


def fetch_recent() -> pd.DataFrame:
    """Pobiera ostatnie 3 dni + 2 dni prognozy z Forecast API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relativehumidity_2m,pressure_msl,"
        f"windspeed_10m,windgusts_10m,winddirection_10m,"
        f"precipitation,cloudcover,shortwave_radiation,direct_radiation,uv_index,"
        f"apparent_temperature,dewpoint_2m,"
        f"temperature_850hPa,windspeed_850hPa,winddirection_850hPa,"
        f"geopotential_height_500hPa"
        f"&past_days=3&forecast_days=2&timezone=UTC"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "timestamp":         pd.to_datetime(h["time"]).tz_localize("UTC"),
        "temperature_c":     pd.array(h["temperature_2m"],      dtype=float),
        "humidity_pct":      pd.array(h["relativehumidity_2m"],  dtype=float),
        "pressure_hpa":      pd.array(h["pressure_msl"],         dtype=float),
        "wind_speed_kmh":    pd.array(h["windspeed_10m"],        dtype=float),
        "wind_gust_kmh":     pd.array(h["windgusts_10m"],        dtype=float),
        "wind_direction":    pd.array(h["winddirection_10m"],    dtype=float),
        "rain_rate_mm":      pd.array(h["precipitation"],        dtype=float),
        "cloudcover_pct":    pd.array(h["cloudcover"],           dtype=float),
        "shortwave_rad":     pd.array(h["shortwave_radiation"],  dtype=float),
        "direct_rad":        pd.array(h["direct_radiation"],     dtype=float),
        "uv_index_nwp":      pd.array(h["uv_index"],             dtype=float),
        "apparent_temp":     pd.array(h["apparent_temperature"], dtype=float),
        "dewpoint":          pd.array(h["dewpoint_2m"],          dtype=float),
        "temp_850hpa":       pd.array(h["temperature_850hPa"],   dtype=float),
        "wind_dir_sin":      np.sin(np.deg2rad(pd.array(h["winddirection_10m"], dtype=float))),
        "wind_dir_cos":      np.cos(np.deg2rad(pd.array(h["winddirection_10m"], dtype=float))),
        "u_wind_850hpa":    -pd.array(h["windspeed_850hPa"], dtype=float) * np.sin(np.deg2rad(pd.array(h["winddirection_850hPa"], dtype=float))),
        "v_wind_850hpa":    -pd.array(h["windspeed_850hPa"], dtype=float) * np.cos(np.deg2rad(pd.array(h["winddirection_850hPa"], dtype=float))),
        "geopotential_500":  pd.array(h["geopotential_height_500hPa"], dtype=float),
    })
    return df


# ─────────────────────────────────────────────
# ESTYMACJA DANYCH CZUJNIKÓW WODNYCH
# Open-Meteo nie ma danych wodnych — estymujemy
# na podstawie znanych korelacji hydrologicznych
# ─────────────────────────────────────────────

def estimate_water_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estymuje wartości czujników wodnych z danych meteorologicznych.
    Modele zastępcze — zostaną zastąpione prawdziwymi danymi gdy stacja powstanie.
    """
    df = df.copy()
    n  = len(df)

    # ── Temperatura wody ─────────────────────────────────────────────────────
    # Woda reaguje wolniej na zmiany temp niż powietrze
    # Przybliżenie: rolling 5-dniowy z opóźnieniem 12h + offset sezonowy
    temp_smooth = df["temperature_c"].rolling(120, min_periods=1).mean().shift(12).ffill()
    doy         = df["timestamp"].dt.day_of_year
    seasonal_offset = 2.0 * np.sin(2 * np.pi * (doy - 30) / 365)  # woda cieplejsza latem
    df["water_temp_c"] = (temp_smooth * 0.85 + df["temperature_c"] * 0.15
                          + seasonal_offset).clip(-2, 35).round(2)

    # ── Poziom wody ──────────────────────────────────────────────────────────
    # Bazowy poziom + kumulatywne opady z ostatnich 72h - evaporacja
    rain_72h      = df["rain_rate_mm"].rolling(72, min_periods=1).sum()
    evap          = (df["temperature_c"].clip(0) * 0.03 +
                     df["shortwave_rad"].fillna(0) * 0.001)
    water_level   = 45.0 + rain_72h * 0.8 - evap.rolling(24, min_periods=1).sum() * 0.5
    df["water_level_cm"] = water_level.clip(5, 500).round(1)

    # ── pH wody ──────────────────────────────────────────────────────────────
    # pH spada po deszczu (kwaśny deszcz), rośnie przy wysokim CO2 (noc)
    rain_effect   = -0.3 * (df["rain_rate_mm"] > 2).astype(float)
    temp_effect   = -0.01 * (df["water_temp_c"] - 15)   # wyższa temp → niższe pH
    hour          = df["timestamp"].dt.hour
    co2_effect    = 0.1 * ((hour < 6) | (hour > 20)).astype(float)  # noc: fotosynteza off
    base_ph       = 7.4
    noise         = np.random.default_rng(42).normal(0, 0.05, n)
    df["water_ph"] = (base_ph + rain_effect + temp_effect + co2_effect + noise).clip(6.0, 9.0).round(2)

    # ── TDS (Total Dissolved Solids) ─────────────────────────────────────────
    # Wyższy po suchych okresach (koncentracja), niższy po deszczach (rozcieńczenie)
    days_without_rain = (df["rain_rate_mm"] < 0.1).astype(int)
    dry_streak        = days_without_rain.rolling(72, min_periods=1).sum()
    dilution          = -50 * (df["rain_rate_mm"] > 5).astype(float)
    noise_tds         = np.random.default_rng(43).normal(0, 10, n)
    df["water_tds"]   = (250 + dry_streak * 2 + dilution + noise_tds).clip(50, 1500).round(0)
    df["water_conductivity"] = (df["water_tds"] / 0.67).round(1)

    # ── Dissolved Oxygen ─────────────────────────────────────────────────────
    # DO spada z temperaturą wody (odwrotna korelacja)
    # DO_sat [mg/L] ≈ 14.62 - 0.3898*T + 0.006969*T² - 0.00005896*T³
    T   = df["water_temp_c"]
    do_sat = (14.62 - 0.3898 * T + 0.006969 * T**2 - 0.00005896 * T**3)
    # Nasycenie: wyższe w dzień (fotosynteza), niższe w nocy
    hour      = df["timestamp"].dt.hour
    photo     = 0.5 * np.sin(np.pi * (hour - 6) / 12).clip(0)
    noise_do  = np.random.default_rng(44).normal(0, 0.2, n)
    df["water_oxygen"] = (do_sat * 0.9 + photo + noise_do).clip(4.0, 16.0).round(2)

    # ── Turbidity (mętność) ──────────────────────────────────────────────────
    # Wzrasta przy intensywnych opadach (spływ), spada po kilku godzinach
    rain_intense  = df["rain_rate_mm"].clip(0, 20)
    turb_spike    = rain_intense * 15
    turb_decay    = turb_spike.ewm(halflife=6).mean()   # zanik przez ~6h
    noise_turb    = np.random.default_rng(45).normal(0, 0.5, n).clip(0)
    df["turbidity_ntu"] = (2.0 + turb_decay + noise_turb).clip(0, 500).round(2)

    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wylicza cechy pochodne wymagane przez modele ML."""
    df = df.copy()

    # Pochodne ciśnienia
    df["d_pressure_3h"]  = df["pressure_hpa"].diff(3).fillna(0).round(2)
    df["d_pressure_24h"] = df["pressure_hpa"].diff(24).fillna(0).round(2)
    df["pressure_trend"] = np.sign(df["d_pressure_3h"]).astype(int)

    # Solar radiation → lux (odwrotność LUX_TO_WATT = 1/0.0079 ≈ 126.6)
    df["lux"]            = (df["shortwave_rad"].fillna(0) * 126.6).clip(0).round(0)

    # UV index z promieniowania (jeśli nie ma z API)
    if "uv_index_nwp" not in df.columns:
        doy      = df["timestamp"].dt.day_of_year
        uv_max   = 2.0 + 5.0 * np.sin(np.pi * (doy - 80) / 185).clip(0)  # sezonowo
        rad_frac = (df["shortwave_rad"].fillna(0) / 800).clip(0, 1)
        df["uv_index"] = (uv_max * rad_frac).clip(0, 12).round(2)
    else:
        df["uv_index"]  = df["uv_index_nwp"].fillna(0).clip(0, 12).round(2)

    # Świt / zmierzch
    def solar_elev(ts, lat=LAT):
        doy       = ts.day_of_year
        hour_utc  = ts.hour + ts.minute / 60
        decl      = math.radians(23.45 * math.sin(math.radians(360/365*(doy-81))))
        ha        = math.radians(15 * (hour_utc - 12))
        lat_r     = math.radians(lat)
        sin_e     = (math.sin(lat_r)*math.sin(decl) +
                     math.cos(lat_r)*math.cos(decl)*math.cos(ha))
        return math.degrees(math.asin(max(-1, min(1, sin_e))))

    elevs          = [solar_elev(ts) for ts in df["timestamp"]]
    df["solar_elev"] = elevs
    df["is_dawn"]  = ((np.array(elevs) > -6) & (np.array(elevs) < 6) &
                      (df["timestamp"].dt.hour < 12)).astype(int)
    df["is_dusk"]  = ((np.array(elevs) > -6) & (np.array(elevs) < 6) &
                      (df["timestamp"].dt.hour >= 12)).astype(int)

    # Pora dnia cyklicznie
    h              = df["timestamp"].dt.hour
    doy            = df["timestamp"].dt.day_of_year
    df["hour_sin"] = np.sin(2 * np.pi * h   / 24).round(4)
    df["hour_cos"] = np.cos(2 * np.pi * h   / 24).round(4)
    df["doy_sin"]  = np.sin(2 * np.pi * doy / 365).round(4)
    df["doy_cos"]  = np.cos(2 * np.pi * doy / 365).round(4)

    # Faza księżyca
    known_new      = datetime(2000, 1, 6, tzinfo=timezone.utc)
    def moon(ts):
        days  = (ts.to_pydatetime() - known_new).days
        phase = (days % 29.53058867) / 29.53058867
        illum = 50 * (1 - math.cos(2 * math.pi * phase))
        return round(phase, 3), round(illum, 1)

    moon_data           = [moon(ts) for ts in df["timestamp"]]
    df["moon_phase"]    = [m[0] for m in moon_data]
    df["moon_illum"]    = [m[1] for m in moon_data]

    # Suma opadów krocząca
    df["precip_7d_sum"] = df["rain_rate_mm"].rolling(168, min_periods=1).sum().round(2)

    return df


# ─────────────────────────────────────────────
# ZAPIS DO INFLUXDB
# ─────────────────────────────────────────────

def write_to_influx(df: pd.DataFrame, client: InfluxDBClient,
                    batch_size: int = 5000):
    """Zapisuje DataFrame do InfluxDB w batchach."""
    write_api = client.write_api(write_options=SYNCHRONOUS)
    total     = 0
    errors    = 0

    # Podziel na batche
    for start in range(0, len(df), batch_size):
        chunk  = df.iloc[start:start + batch_size]
        points = []

        for _, row in chunk.iterrows():
            ts = row["timestamp"]
            if pd.isna(ts):
                continue

            # ── Measurement: atmosphere ───────────────────────────────────
            atm = (
                Point(MEAS_ATMOSPHERE)
                .tag("location", LOCATION)
                .tag("source", "openmeteo_simulated")
                .time(ts, WritePrecision.S)
            )
            atm_fields = {
                "temperature_c":   "temperature_c",
                "humidity_pct":    "humidity_pct",
                "pressure_hpa":    "pressure_hpa",
                "wind_speed_kmh":  "wind_speed_kmh",
                "wind_gust_kmh":   "wind_gust_kmh",
                "wind_direction":  "wind_direction",
                "rain_rate_mm":    "rain_rate_mm",
                "cloudcover_pct":  "cloudcover_pct",
                "shortwave_rad":   "shortwave_rad",
                "lux":             "lux",
                "uv_index":        "uv_index",
                "solar_elev":      "solar_elev",
                "apparent_temp":   "apparent_temp",
                "dewpoint":        "dewpoint",
                "temp_850hpa":     "temp_850hpa",
                "wind_dir_sin":    "wind_dir_sin",
                "wind_dir_cos":    "wind_dir_cos",
                "u_wind_850hpa":   "u_wind_850hpa",
                "v_wind_850hpa":   "v_wind_850hpa",
                "geopotential_500":"geopotential_500",
            }
            for df_col, influx_field in atm_fields.items():
                if df_col in row.index and not pd.isna(row[df_col]):
                    atm.field(influx_field, float(row[df_col]))
            points.append(atm)

            # ── Measurement: water ────────────────────────────────────────
            wat = (
                Point(MEAS_WATER)
                .tag("location", LOCATION)
                .tag("source", "estimated")   # ← zmień na "sensor" gdy stacja gotowa
                .time(ts, WritePrecision.S)
            )
            water_fields = {
                "water_temp_c":      "temperature_c",
                "water_level_cm":    "level_cm",
                "water_ph":          "ph",
                "water_tds":         "tds",
                "water_conductivity":"conductivity",
                "water_oxygen":      "dissolved_oxygen",
                "turbidity_ntu":     "turbidity_ntu",
            }
            for df_col, influx_field in water_fields.items():
                if df_col in row.index and not pd.isna(row[df_col]):
                    wat.field(influx_field, float(row[df_col]))
            points.append(wat)

            # ── Measurement: derived ──────────────────────────────────────
            drv = (
                Point(MEAS_DERIVED)
                .tag("location", LOCATION)
                .time(ts, WritePrecision.S)
            )
            derived_fields = {
                "d_pressure_3h":  "d_pressure_3h",
                "d_pressure_24h": "d_pressure_24h",
                "pressure_trend": "pressure_trend",
                "hour_sin":       "hour_sin",
                "hour_cos":       "hour_cos",
                "doy_sin":        "doy_sin",
                "doy_cos":        "doy_cos",
                "is_dawn":        "is_dawn",
                "is_dusk":        "is_dusk",
                "moon_phase":     "moon_phase",
                "moon_illum":     "moon_illumination",
                "precip_7d_sum":  "precip_7d_sum",
            }
            for df_col, influx_field in derived_fields.items():
                if df_col in row.index and not pd.isna(row[df_col]):
                    drv.field(influx_field, float(row[df_col]))
            points.append(drv)

        try:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)
            total += len(chunk)
            print(f"  Zapisano {total}/{len(df)} rekordów...", end="\r", flush=True)
        except Exception as e:
            errors += 1
            print(f"\n  Błąd zapisu batch {start}-{start+batch_size}: {e}")

    print(f"\n  Łącznie: {total} rekordów, {errors} błędów")
    return total


# ─────────────────────────────────────────────
# GŁÓWNE FUNKCJE
# ─────────────────────────────────────────────

def process_chunk(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Pipeline: surowe dane → czujniki wodne → cechy pochodne."""
    df = df_raw.copy()
    df[["temperature_c", "humidity_pct", "pressure_hpa",
        "wind_speed_kmh", "wind_gust_kmh", "rain_rate_mm",
        "shortwave_rad"]] = df[["temperature_c", "humidity_pct", "pressure_hpa",
                                 "wind_speed_kmh", "wind_gust_kmh", "rain_rate_mm",
                                 "shortwave_rad"]].interpolate().ffill().bfill()
    df = estimate_water_sensors(df)
    df = compute_derived_features(df)
    return df


def backfill(years: int = 5, days: int = None):
    """Pobierz i zapisz dane historyczne."""
    today    = date.today()
    if days:
        start_dt = today - timedelta(days=days)
    else:
        start_dt = date(today.year - years, 1, 1)

    end_dt   = today - timedelta(days=1)   # archive do wczoraj
    total_days = (end_dt - start_dt).days

    print(f"[{datetime.now():%H:%M}] Backfill: {start_dt} → {end_dt} ({total_days} dni)")

    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    current = start_dt
    written = 0
    while current <= end_dt:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end_dt)
        print(f"  Pobieram {current} → {chunk_end}...", end=" ", flush=True)

        try:
            df_raw = fetch_archive_chunk(str(current), str(chunk_end))
            df     = process_chunk(df_raw)
            print(f"{len(df)} godz →", end=" ", flush=True)
            written += write_to_influx(df, client)
        except Exception as e:
            print(f"BŁĄD: {e}")

        current = chunk_end + timedelta(days=1)
        time.sleep(1)  # grzeczność wobec API

    client.close()
    print(f"\nBackfill zakończony. Zapisano {written} rekordów.")


def update_recent():
    """Pobierz ostatnie 3 dni + prognozę i zapisz."""
    print(f"[{datetime.now():%H:%M}] Aktualizacja ostatnich 3 dni + prognoza...")
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    try:
        df_raw = fetch_recent()
        df     = process_chunk(df_raw)
        print(f"  Pobrano {len(df)} rekordów")
        write_to_influx(df, client)
    except Exception as e:
        print(f"BŁĄD: {e}")
    finally:
        client.close()


def daemon_loop(interval_minutes: int = 60):
    """Daemon — aktualizuje dane co `interval_minutes` minut."""
    print(f"[{datetime.now():%H:%M}] Daemon uruchomiony (co {interval_minutes} min)")
    while True:
        try:
            update_recent()
        except Exception as e:
            print(f"Błąd w pętli daemon: {e}")
        next_run = datetime.now() + timedelta(minutes=interval_minutes)
        print(f"  Następna aktualizacja: {next_run:%H:%M}")
        time.sleep(interval_minutes * 60)


# ─────────────────────────────────────────────
# WERYFIKACJA POŁĄCZENIA
# ─────────────────────────────────────────────

def verify_connection():
    """Sprawdź połączenie z InfluxDB i wylistuj buckety."""
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        health = client.health()
        print(f"InfluxDB: {health.status} ({INFLUX_URL})")
        buckets = client.buckets_api().find_buckets().buckets
        names   = [b.name for b in buckets]
        print(f"Buckety: {names}")
        if INFLUX_BUCKET not in names:
            print(f"  ⚠ Bucket '{INFLUX_BUCKET}' nie istnieje — utwórz go w InfluxDB UI")
        else:
            print(f"  ✓ Bucket '{INFLUX_BUCKET}' gotowy")
        client.close()
        return True
    except Exception as e:
        print(f"Błąd połączenia z InfluxDB: {e}")
        print(f"Sprawdź: URL={INFLUX_URL}, token, org, czy InfluxDB działa")
        return False


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-Meteo → InfluxDB loader")
    parser.add_argument("--backfill", action="store_true",
                        help="Backfill historyczny")
    parser.add_argument("--recent",   action="store_true",
                        help="Ostatnie 3 dni + prognoza")
    parser.add_argument("--daemon",   action="store_true",
                        help="Daemon — odświeżaj co godzinę")
    parser.add_argument("--verify",   action="store_true",
                        help="Sprawdź połączenie z InfluxDB")
    parser.add_argument("--years",    type=int, default=5,
                        help="Ile lat backfill (domyślnie 5)")
    parser.add_argument("--days",     type=int, default=None,
                        help="Ile dni backfill (zamiast --years)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Interwał daemon w minutach (domyślnie 60)")
    args = parser.parse_args()

    if args.verify:
        verify_connection()

    elif args.backfill:
        if verify_connection():
            backfill(years=args.years, days=args.days)

    elif args.recent:
        if verify_connection():
            update_recent()

    elif args.daemon:
        if verify_connection():
            daemon_loop(interval_minutes=args.interval)

    else:
        parser.print_help()
        print("\nPrzykłady:")
        print("  python openmeteo_to_influx.py --verify")
        print("  python openmeteo_to_influx.py --backfill --years 5")
        print("  python openmeteo_to_influx.py --backfill --days 30")
        print("  python openmeteo_to_influx.py --recent")
        print("  python openmeteo_to_influx.py --daemon --interval 60")