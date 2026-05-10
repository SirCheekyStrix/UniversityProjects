"""
influx_weather.py — pipeline predykcji dla wszystkich zbiorników z API.

Dla każdego zbiornika:
1. Pobierz lokalizację z /api/reservoirs/{id}
2. Sprawdź czy w InfluxDB (measurements / historical_data) są dane w promieniu 25km
3. Jeśli TAK → użyj tych danych jako input
4. Jeśli NIE → pobierz dane z Open-Meteo dla tej lokalizacji
5. Uruchom TCN / LightGBM / TFT → zapisz do bucket: predictions

Użycie:
    python influx_weather.py --all              # wszystkie zbiorniki
    python influx_weather.py --id 1             # jeden zbiornik
    python influx_weather.py --all --models tcn,fish
    python influx_weather.py --list             # wylistuj dostępne zbiorniki
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────
API_BASE       = "http://localhost:8080/api"

# Credentials do JWT (ustaw przez env lub --username/--password)
API_USERNAME   = os.environ.get("API_USERNAME", "admin")
API_PASSWORD   = os.environ.get("API_PASSWORD", "")
API_LOGIN_URL  = os.environ.get("API_LOGIN_URL",   f"{API_BASE.replace('/api','')}/auth/login")
API_REFRESH_URL= os.environ.get("API_REFRESH_URL", f"{API_BASE.replace('/api','')}/auth/refresh")

INFLUX_URL     = "http://localhost:8086"
INFLUX_TOKEN   = "mH1sJUpajjdlKcQrM64YVu8efBZymCm--X0Jp2nRoaHFIZduitvapbZATXrA6t2TxnwQ2EVJ8RuxrfNM4efeDA=="
INFLUX_ORG     = "zlote-branie"
BUCKET_DATA    = ["measurements", "historical_data"]   # szukaj danych w obu
BUCKET_PRED    = "predictions"

SEARCH_RADIUS_KM = 25.0    # promień szukania danych w InfluxDB
MODELS_DEFAULT   = ["tcn", "fish", "tft"]

logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(asctime)s] %(levelname)s %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# POMOCNICZE
# ─────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Odległość w km między dwoma punktami GPS."""
    R   = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ  = math.radians(lat2 - lat1)
    dλ  = math.radians(lon2 - lon1)
    a   = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def get_tm(username: str = None, password: str = None) -> "TokenManager":
    """Zwraca singleton TokenManager z automatycznym odświeżaniem JWT."""
    from token_manager import get_token_manager
    return get_token_manager(
        login_url    = API_LOGIN_URL,
        refresh_url  = API_REFRESH_URL,
        username     = username or API_USERNAME,
        password     = password or API_PASSWORD,
    )


# ─────────────────────────────────────────────
# POBIERANIE ZBIORNIKÓW Z API
# ─────────────────────────────────────────────

def fetch_all_reservoirs() -> list[dict]:
    """Pobierz wszystkie zbiorniki z /api/reservoirs."""
    url = f"{API_BASE}/reservoirs"
    try:
        tm = get_tm()
        r  = tm.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        # Obsłuż zarówno listę jak i {"content": [...]} (Spring Page)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("content", "data", "reservoirs", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        log.warning(f"Nieoczekiwana struktura /api/reservoirs: {type(data)}")
        return []
    except Exception as e:
        log.error(f"Błąd pobierania listy zbiorników: {e}")
        return []


def fetch_reservoir(reservoir_id: int) -> Optional[dict]:
    """Pobierz jeden zbiornik z /api/reservoirs/{id}."""
    url = f"{API_BASE}/reservoirs/{reservoir_id}"
    try:
        tm = get_tm()
        r  = tm.get(url, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"  Błąd reservoir {reservoir_id}: {e}")
        return None


def iter_reservoirs(max_id: int = 200) -> list[dict]:
    """
    Iteruje po ID 1..max_id gdy /api/reservoirs nie jest dostępne.
    Zatrzymuje się po 10 kolejnych 404.
    """
    results   = []
    misses    = 0
    for rid in range(1, max_id + 1):
        res = fetch_reservoir(rid)
        if res is None:
            misses += 1
            if misses >= 10:
                break
            continue
        misses = 0
        results.append(res)
    return results


# ─────────────────────────────────────────────
# SPRAWDZANIE DANYCH W INFLUXDB
# ─────────────────────────────────────────────

def find_nearest_influx_location(lat: float, lon: float,
                                  radius_km: float = SEARCH_RADIUS_KM) -> Optional[dict]:
    """
    Sprawdza czy w InfluxDB są dane pogodowe dla lokalizacji w promieniu radius_km.
    Szuka tagu 'location' lub pola lat/lon w obu bucketach.
    Zwraca {"bucket": ..., "location": ..., "distance_km": ..., "lat": ..., "lon": ...}
    lub None.
    """
    try:
        from influxdb_client import InfluxDBClient
    except ImportError:
        log.warning("influxdb-client nie zainstalowany — pomijam sprawdzanie InfluxDB")
        return None

    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()

    best = None

    for bucket in BUCKET_DATA:
        # Pobierz unikalne tagi 'location' z ostatnich 7 dni
        flux = f'''
import "influxdata/influxdb/schema"
schema.tagValues(
  bucket: "{bucket}",
  tag: "location",
  start: -7d
)
'''
        try:
            tables = query_api.query(flux)
            locations = [row.get_value() for table in tables for row in table.records]
        except Exception:
            locations = []

        for loc_tag in locations:
            # Spróbuj zdekodować lat/lon z tagu (format: "lat_lon" lub "name")
            coords = _tag_to_coords(loc_tag, query_api, bucket)
            if coords is None:
                continue
            dist = haversine_km(lat, lon, coords["lat"], coords["lon"])
            if dist <= radius_km:
                if best is None or dist < best["distance_km"]:
                    best = {
                        "bucket":      bucket,
                        "location":    loc_tag,
                        "distance_km": round(dist, 2),
                        "lat":         coords["lat"],
                        "lon":         coords["lon"],
                    }

    client.close()
    return best


def _tag_to_coords(location_tag: str, query_api, bucket: str) -> Optional[dict]:
    """
    Próbuje odczytać lat/lon z pól w InfluxDB dla danego location tagu.
    Zakłada że openmeteo_to_influx.py nie zapisuje coords jako pól — 
    więc dekodujemy z nazwy tagu (format "lat53.01_lon18.59") lub z metadanych.
    """
    # Format 1: "lat53.0138_lon18.5981"
    if location_tag.startswith("lat") and "_lon" in location_tag:
        try:
            parts = location_tag.replace("lat", "").split("_lon")
            return {"lat": float(parts[0]), "lon": float(parts[1])}
        except Exception:
            pass

    # Format 2: znana lokalizacja po nazwie (słownik)
    known = {
        "torun":     {"lat": 53.0138, "lon": 18.5981},
        "bydgoszcz": {"lat": 53.1235, "lon": 18.0084},
        "gdansk":    {"lat": 54.3520, "lon": 18.6466},
        "warszawa":  {"lat": 52.2297, "lon": 21.0122},
        "poznan":    {"lat": 52.4064, "lon": 16.9252},
        "wroclaw":   {"lat": 51.1079, "lon": 17.0385},
    }
    name_lower = location_tag.lower().replace("-", "").replace("_", "")
    for key, coords in known.items():
        if key in name_lower:
            return coords

    # Format 3: zapytaj InfluxDB o pole lat/lon jeśli zapisane
    try:
        flux = f'''
from(bucket: "{bucket}")
  |> range(start: -7d)
  |> filter(fn: (r) => r["location"] == "{location_tag}")
  |> filter(fn: (r) => r["_field"] == "lat" or r["_field"] == "lon")
  |> last()
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> limit(n: 1)
'''
        tables = query_api.query(flux)
        for table in tables:
            for row in table.records:
                lat = row.values.get("lat")
                lon = row.values.get("lon")
                if lat and lon:
                    return {"lat": float(lat), "lon": float(lon)}
    except Exception:
        pass

    return None


def fetch_influx_weather(location_tag: str, bucket: str,
                          hours_back: int = 72) -> Optional[pd.DataFrame]:
    """Pobiera dane pogodowe z InfluxDB dla danej lokalizacji."""
    try:
        from influxdb_client import InfluxDBClient
        client    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()

        flux = f'''
from(bucket: "{bucket}")
  |> range(start: -{hours_back}h)
  |> filter(fn: (r) => r["location"] == "{location_tag}")
  |> filter(fn: (r) => r["_measurement"] == "atmosphere" or r["_measurement"] == "derived")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
'''
        tables = query_api.query(flux)
        rows   = []
        for table in tables:
            for row in table.records:
                rows.append(row.values)

        client.close()
        if not rows:
            return None

        df = pd.DataFrame(rows)
        df = df.rename(columns={"_time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)
        log.info(f"  InfluxDB: {len(df)} rekordów z bucket={bucket}, location={location_tag}")
        return df

    except Exception as e:
        log.error(f"  Błąd InfluxDB fetch: {e}")
        return None


# ─────────────────────────────────────────────
# POBIERANIE DANYCH Z OPEN-METEO
# ─────────────────────────────────────────────

def fetch_openmeteo(lat: float, lon: float, past_days: int = 7) -> pd.DataFrame:
    """Pobiera dane pogodowe z Open-Meteo Forecast API dla podanych współrzędnych."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relativehumidity_2m,pressure_msl,"
        f"windspeed_10m,windgusts_10m,winddirection_10m,precipitation,"
        f"cloudcover,shortwave_radiation,apparent_temperature,dewpoint_2m,"
        f"temperature_850hPa,windspeed_850hPa,winddirection_850hPa,"
        f"geopotential_height_500hPa,uv_index"
        f"&past_days={past_days}&forecast_days=2&timezone=UTC"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    h   = r.json()["hourly"]
    n   = len(h["time"])
    ws850 = np.array(h.get("windspeed_850hPa",   [0]*n), dtype=float)
    wd850 = np.array(h.get("winddirection_850hPa",[0]*n), dtype=float)
    wd10  = np.array(h.get("winddirection_10m",   [0]*n), dtype=float)

    df = pd.DataFrame({
        "timestamp":           pd.to_datetime(h["time"]),
        "temperature":         h["temperature_2m"],
        "humidity":            h["relativehumidity_2m"],
        "pressure":            h["pressure_msl"],
        "wind_speed":          h["windspeed_10m"],
        "wind_gust":           h["windgusts_10m"],
        "rain_rate":           h["precipitation"],
        "wind_dir_sin":        np.sin(np.deg2rad(wd10)),
        "wind_dir_cos":        np.cos(np.deg2rad(wd10)),
        "temp_850hpa":         h.get("temperature_850hPa",        [0]*n),
        "u_wind_850hpa":       -ws850 * np.sin(np.deg2rad(wd850)),
        "v_wind_850hpa":       -ws850 * np.cos(np.deg2rad(wd850)),
        "geopotential_500":    h.get("geopotential_height_500hPa", [0]*n),
        "cloudcover":          h["cloudcover"],
        "uv_index":            [v or 0 for v in h.get("uv_index", [0]*n)],
        "apparent_temp":       h["apparent_temperature"],
        "dewpoint":            h["dewpoint_2m"],
        "shortwave_radiation": h.get("shortwave_radiation",        [0]*n),
    })
    df = df.interpolate().ffill().bfill()
    log.info(f"  Open-Meteo: {len(df)} godz dla ({lat:.4f}, {lon:.4f})")
    return df


# ─────────────────────────────────────────────
# PREDYKCJA DLA JEDNEJ LOKALIZACJI
# ─────────────────────────────────────────────

def run_predictions(reservoir: dict, df_weather: pd.DataFrame,
                    models: list[str]) -> dict:
    """
    Uruchamia wybrane modele dla danego zbiornika i zwraca słownik wyników.
    df_weather: DataFrame z danymi pogodowymi (z InfluxDB lub Open-Meteo)
    """
    results = {}
    rid     = reservoir["id"]
    name    = reservoir.get("name", f"reservoir_{rid}")

    # ── TCN — pogoda 24h ─────────────────────────────────────────────────────
    if "tcn" in models:
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from weather_model import add_features, FEATURES_IN, TARGETS
            from weather_model import load_model, load_scalers, predict as tcn_predict_fn

            df_feat = add_features(df_weather.copy())

            # Sprawdź czy wszystkie features dostępne
            missing = [f for f in FEATURES_IN if f not in df_feat.columns]
            if missing:
                log.warning(f"  TCN: brak cech {missing} — uzupełniam zerami")
                for f in missing:
                    df_feat[f] = 0.0

            result_tcn = tcn_predict_fn(df_feat)
            result_tcn["reservoir_id"]   = rid
            result_tcn["reservoir_name"] = name
            results["tcn"] = result_tcn
            log.info(f"  TCN: OK ({len(result_tcn.get('forecast', []))} godz)")
        except Exception as e:
            log.error(f"  TCN błąd: {e}")

    # ── LightGBM — ryby 24h ──────────────────────────────────────────────────
    if "fish" in models:
        try:
            from fish_model import SPECIES, add_fish_features
            from fish_predict import predict_24h as fish_predict_fn
            import lightgbm as lgb

            df_fish = add_fish_features(df_weather.copy())
            result_fish = fish_predict_fn(df_input=df_fish)
            result_fish["reservoir_id"]   = rid
            result_fish["reservoir_name"] = name
            results["fish"] = result_fish
            log.info(f"  Fish: OK")
        except Exception as e:
            log.error(f"  Fish błąd: {e}")

    # ── TFT — pogoda 10 dni ──────────────────────────────────────────────────
    if "tft" in models:
        try:
            from tft_model import add_features as tft_add_features
            from tft_predict import predict_10days as tft_predict_fn

            result_tft = tft_predict_fn(df_input=df_weather.copy())
            result_tft["reservoir_id"]   = rid
            result_tft["reservoir_name"] = name
            results["tft"] = result_tft
            log.info(f"  TFT: OK ({len(result_tft.get('forecast', []))} dni)")
        except Exception as e:
            log.error(f"  TFT błąd: {e}")

    return results


# ─────────────────────────────────────────────
# ZAPIS DO INFLUXDB PREDICTIONS
# ─────────────────────────────────────────────

def write_predictions(reservoir: dict, results: dict):
    """Zapisuje wyniki predykcji do bucket predictions z tagiem reservoir_id."""
    try:
        from influxdb_client import InfluxDBClient, Point
        from influxdb_client.client.write_api import SYNCHRONOUS
    except ImportError:
        log.error("influxdb-client nie zainstalowany")
        return

    rid    = reservoir["id"]
    name   = reservoir.get("name", f"reservoir_{rid}")
    lat    = reservoir.get("latitude",  0.0)
    lon    = reservoir.get("longitude", 0.0)
    owner  = reservoir.get("owner",     "unknown")
    gen_at = datetime.now(timezone.utc).isoformat()

    client    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    points    = []

    # ── TCN ──────────────────────────────────────────────────────────────────
    if "tcn" in results:
        result = results["tcn"]
        for row in result.get("forecast", []):
            try:
                ts = datetime.fromisoformat(
                    row["timestamp"].replace("Z", "+00:00")
                ).replace(tzinfo=timezone.utc)
            except Exception:
                continue
            p = (
                Point("forecast_weather_24h")
                .tag("reservoir_id",   str(rid))
                .tag("reservoir_name", name)
                .tag("owner",          owner)
                .tag("model",          "TCN")
                .tag("lat",            f"{lat:.4f}")
                .tag("lon",            f"{lon:.4f}")
                .tag("generated_at",   gen_at[:19])
                .time(ts)
            )
            for field, val in row.items():
                if field == "timestamp":
                    continue
                try:
                    p.field(field, float(val))
                except (TypeError, ValueError):
                    pass
            points.append(p)

    # ── Fish ─────────────────────────────────────────────────────────────────
    if "fish" in results:
        result  = results["fish"]
        species = result.get("species", [])
        for row in result.get("forecast", []):
            try:
                ts = datetime.fromisoformat(
                    row["timestamp"].replace("Z", "+00:00")
                ).replace(tzinfo=timezone.utc)
            except Exception:
                continue
            p = (
                Point("forecast_fish_24h")
                .tag("reservoir_id",   str(rid))
                .tag("reservoir_name", name)
                .tag("owner",          owner)
                .tag("model",          "LightGBM")
                .tag("lat",            f"{lat:.4f}")
                .tag("lon",            f"{lon:.4f}")
                .tag("generated_at",   gen_at[:19])
                .time(ts)
            )
            for sp in species:
                if sp in row:
                    p.field(f"prob_{sp}", float(row[sp]))
            for field in ["temperature", "pressure", "wind_speed",
                          "pop_pct", "moon_phase", "is_dawn", "is_dusk"]:
                if field in row:
                    try:
                        p.field(field, float(row[field]))
                    except (TypeError, ValueError):
                        pass
            points.append(p)

    # ── TFT ──────────────────────────────────────────────────────────────────
    if "tft" in results:
        result = results["tft"]
        for day in result.get("forecast", []):
            try:
                ts = datetime.fromisoformat(day["date"]).replace(
                    hour=12, tzinfo=timezone.utc
                )
            except Exception:
                continue
            p = (
                Point("forecast_weather_10d")
                .tag("reservoir_id",   str(rid))
                .tag("reservoir_name", name)
                .tag("owner",          owner)
                .tag("model",          "TFT")
                .tag("day",            str(day.get("day", 0)))
                .tag("lat",            f"{lat:.4f}")
                .tag("lon",            f"{lon:.4f}")
                .tag("generated_at",   gen_at[:19])
                .time(ts)
            )
            if "pop_pct" in day:
                p.field("pop_pct", float(day["pop_pct"]))
            for target in ["temp_max","temp_min","precip_sum","wind_max",
                           "wind_mean","pressure_mean","cloudcover_mean","humidity_mean"]:
                if target not in day:
                    continue
                vals = day[target]
                if isinstance(vals, dict):
                    for qkey, suffix in [("q10","_q10"),("median",""),("q90","_q90"),
                                         ("nwp","_nwp"),("tft","_tft_raw")]:
                        if vals.get(qkey) is not None:
                            p.field(f"{target}{suffix}", float(vals[qkey]))
                else:
                    p.field(target, float(vals))
            points.append(p)

    # ── Zapis ─────────────────────────────────────────────────────────────────
    if points:
        try:
            write_api.write(bucket=BUCKET_PRED, org=INFLUX_ORG, record=points)
            log.info(f"  → InfluxDB predictions: {len(points)} pkt "
                     f"(res={rid} '{name}')")
        except Exception as e:
            log.error(f"  Błąd zapisu predictions: {e}")
    else:
        log.warning(f"  Brak punktów do zapisu dla reservoir {rid}")

    client.close()


# ─────────────────────────────────────────────
# PIPELINE DLA JEDNEGO ZBIORNIKA
# ─────────────────────────────────────────────

def process_reservoir(reservoir: dict, models: list[str]):
    rid  = reservoir["id"]
    name = reservoir.get("name", f"id={rid}")
    lat  = reservoir.get("latitude")
    lon  = reservoir.get("longitude")

    if lat is None or lon is None:
        log.warning(f"[{rid}] '{name}' — brak lat/lon, pomijam")
        return

    log.info(f"[{rid}] '{name}' ({lat:.5f}, {lon:.5f})")

    # 1. Szukaj danych w InfluxDB w promieniu 25km
    nearby = find_nearest_influx_location(lat, lon)

    if nearby:
        dist = nearby["distance_km"]
        log.info(f"  Dane InfluxDB: bucket={nearby['bucket']}, "
                 f"location={nearby['location']}, odległość={dist}km")
        df_weather = fetch_influx_weather(nearby["location"], nearby["bucket"])
        source = f"influxdb:{nearby['bucket']}:{nearby['location']}"
    else:
        log.info(f"  Brak danych InfluxDB w promieniu {SEARCH_RADIUS_KM}km "
                 f"→ pobieram Open-Meteo")
        df_weather = None
        source     = "openmeteo"

    # 2. Fallback do Open-Meteo jeśli brak danych lub za mało
    if df_weather is None or len(df_weather) < 48:
        df_weather = fetch_openmeteo(lat, lon)
        source     = "openmeteo"

    if df_weather is None or df_weather.empty:
        log.error(f"  Brak danych pogodowych dla reservoir {rid} — pomijam")
        return

    log.info(f"  Źródło: {source}, {len(df_weather)} rekordów")

    # 3. Predykcje
    results = run_predictions(reservoir, df_weather, models)
    if not results:
        log.warning(f"  Brak wyników predykcji dla reservoir {rid}")
        return

    # 4. Zapis do InfluxDB
    write_predictions(reservoir, results)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-reservoir prediction pipeline")
    parser.add_argument("--all",      action="store_true",
                        help="Wszystkie zbiorniki")
    parser.add_argument("--id",       type=int, default=None,
                        help="Jeden zbiornik po ID")
    parser.add_argument("--ids",      type=str, default=None,
                        help="Lista ID rozdzielona przecinkami (np. 1,2,5)")
    parser.add_argument("--list",     action="store_true",
                        help="Wylistuj dostępne zbiorniki")
    parser.add_argument("--models",   type=str, default="tcn,fish,tft",
                        help="Modele do uruchomienia (tcn,fish,tft)")
    parser.add_argument("--username", type=str, default=API_USERNAME,
                        help="Username do API (lub ustaw API_USERNAME env)")
    parser.add_argument("--password", type=str, default=None,
                        help="Password do API (lub ustaw API_PASSWORD env)")
    parser.add_argument("--iterate",  action="store_true",
                        help="Iteruj po ID zamiast pobierać listę")
    parser.add_argument("--max-id",   type=int, default=200,
                        help="Maks ID przy --iterate (domyślnie 200)")
    parser.add_argument("--delay",    type=float, default=1.0,
                        help="Opóźnienie między zbiornikami w sek (domyślnie 1s)")
    args = parser.parse_args()

    # Inicjalizuj TokenManager z podanymi credentials
    if args.password:
        os.environ["API_PASSWORD"] = args.password
    if args.username:
        os.environ["API_USERNAME"] = args.username

    # Weryfikacja — pobierz token przy starcie
    tm = get_tm(username=args.username, password=args.password)
    if not tm.get_token():
        log.error("Nie udało się pobrać tokenu — sprawdź API_USERNAME i API_PASSWORD")
        sys.exit(1)
    log.info(f"Token JWT aktywny ({tm.token_info()['access_expires_in']}s)")

    models = [m.strip().lower() for m in args.models.split(",")]

    # ── Zbierz listę zbiorników ───────────────────────────────────────────────
    if args.id:
        res = fetch_reservoir(args.id)
        reservoirs = [res] if res else []

    elif args.ids:
        id_list    = [int(x.strip()) for x in args.ids.split(",")]
        reservoirs = [r for r in (fetch_reservoir(i) for i in id_list)
                      if r is not None]

    elif args.iterate:
        log.info(f"Iteruję po ID 1..{args.max_id}...")
        reservoirs = iter_reservoirs(max_id=args.max_id)

    else:  # --all lub --list
        log.info("Pobieram listę zbiorników...")
        reservoirs = fetch_all_reservoirs()
        if not reservoirs and not args.list:
            log.warning("Endpoint /api/reservoirs niedostępny — próbuję iterację po ID")
            reservoirs = iter_reservoirs()

    if not reservoirs:
        log.error("Brak zbiorników do przetworzenia")
        sys.exit(1)

    # ── Lista ─────────────────────────────────────────────────────────────────
    if args.list:
        print(f"\n{'ID':<6} {'Nazwa':<30} {'Właściciel':<15} {'Lat':>10} {'Lon':>10}")
        print("─" * 75)
        for r in reservoirs:
            print(f"{r['id']:<6} {r.get('name','?'):<30} "
                  f"{r.get('owner','?'):<15} "
                  f"{r.get('latitude',0):>10.5f} {r.get('longitude',0):>10.5f}")
        print(f"\nŁącznie: {len(reservoirs)} zbiorników")
        return

    # ── Pipeline ──────────────────────────────────────────────────────────────
    log.info(f"Zbiorniki: {len(reservoirs)}, modele: {models}")
    ok, fail = 0, 0

    for i, reservoir in enumerate(reservoirs, 1):
        log.info(f"\n── {i}/{len(reservoirs)} ──────────────────────────────")
        try:
            process_reservoir(reservoir, models)
            ok += 1
        except Exception as e:
            log.error(f"  Nieoczekiwany błąd: {e}")
            fail += 1
        if i < len(reservoirs):
            time.sleep(args.delay)

    log.info(f"\n✓ Zakończono: {ok} OK, {fail} błędów")


if __name__ == "__main__":
    main()