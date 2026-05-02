import math
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
from datetime import timezone
import openmeteo_requests
import requests_cache
from retry_requests import retry

# -------------------- Setup Open-Meteo client --------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# -------------------- Helper functions --------------------
def calculate_moon_phase(ts: datetime):
    known_new_moon = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
    synodic_month = 29.53058867
    days = (ts - known_new_moon).total_seconds() / 86400
    phase = (days % synodic_month) / synodic_month
    return {
        "moon_phase": round(phase, 3),
        "moon_illumination": round(0.5 * (1 - math.cos(2 * math.pi * phase)) * 100, 1)
    }

def generate_water(weather):
    air_temp = weather["atm_temperature_C"]
    rain = weather["rain_rate"]
    water_temp = air_temp - 2 + random.uniform(-0.4, 0.4)
    return {
        "water_temperature_C": round(water_temp, 2),
        "water_level_cm": round(120 + rain * 3 + random.uniform(-5, 5), 1),
        "water_pH": round(7.0 + random.uniform(-0.25, 0.25), 2),
        "water_conductivity": round(420 + random.uniform(-40, 40), 1),
        "water_oxygen": round(10.5 - water_temp * 0.15 + random.uniform(-0.3, 0.3), 2)
    }

def generate_system(solar_radiation):
    battery = 3.3 + min(solar_radiation / 1200, 0.6)
    return {
        "battery_voltage": round(min(battery, 4.1), 2),
        "solar_power": round(solar_radiation * 0.02, 2),
        "signal_strength": random.randint(-85, -60)
    }

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calculate_bite_probability(record):
    score = 0.0
    wt = record["water_temperature_C"]
    score += -0.04 * (wt - 15) ** 2 + 2.5
    pressure = record["pressure_hPa"]
    score += (1015 - pressure) * 0.03
    score += math.cos(record["moon_phase"] * 2 * math.pi) * 1.5
    score -= record["wind_speed"] * 0.15
    score -= record["rain_rate"] * 0.3
    prob = sigmoid(score) * 100
    return round(min(max(prob, 0), 100), 1)

def generate_record(ts, weather_row, lat, lon):
    weather = {
        "atm_temperature_C": weather_row["temperature_2m"],
        "humidity_percent": weather_row.get("relative_humidity_2m", 50),
        "pressure_hPa": weather_row.get("surface_pressure", 1013),
        "wind_speed": weather_row.get("wind_speed_10m", 0),
        "wind_direction": weather_row.get("wind_direction_10m", 0),
        "wind_gust": weather_row.get("wind_gusts_10m", 0),
        "rain_rate": weather_row.get("precipitation", 0),
        "solar_radiation": weather_row.get("shortwave_radiation", 0)
    }

    moon = calculate_moon_phase(ts)
    water = generate_water(weather)
    system = generate_system(weather["solar_radiation"])

    rec = {
        "timestamp": ts.isoformat(),
        "latitude": lat,
        "longitude": lon,
        **weather,
        **moon,
        **water,
        **system
    }
    rec["bite_probability"] = calculate_bite_probability(rec)
    return rec

# -------------------- Parquet save --------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PARQUET_FILE = DATA_DIR / "station_timeseries.parquet"

def save_records(records: list[dict]):
    if not records:
        print("⚠️ Brak rekordów – pomijam zapis")
        return

    df_new = pd.DataFrame(records)

    if df_new.empty:
        print("⚠️ DataFrame pusty – pomijam zapis")
        return

    if PARQUET_FILE.exists():
        try:
            old = pd.read_parquet(PARQUET_FILE)

            if not old.empty:
                df = pd.concat([old, df_new], ignore_index=True)
            else:
                df = df_new

        except Exception as e:
            print("⚠️ Błąd odczytu starego parquet, nadpisuję:", e)
            df = df_new
    else:
        df = df_new

    df.to_parquet(PARQUET_FILE, index=False)
    print(f"✅ Zapisano {len(df_new)} rekordów")


# -------------------- Fetch weather via Open-Meteo --------------------
def fetch_weather_open_meteo(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "shortwave_radiation_sum",
        "hourly": ["temperature_2m","relative_humidity_2m","surface_pressure",
                   "wind_speed_10m","wind_direction_10m","wind_gusts_10m","precipitation"],
        "timezone": "auto",
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]  # pierwsza lokalizacja
    hourly = response.Hourly()
    data = {}
    for i, var in enumerate(["temperature_2m","relative_humidity_2m","surface_pressure",
                             "wind_speed_10m","wind_direction_10m","wind_gusts_10m","precipitation"]):
        data[var] = hourly.Variables(i).ValuesAsNumpy()
    data["time"] = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    return data

# -------------------- Main --------------------
def main():
    lat, lon = 53.0138, 18.5981
    start, end = "2024-01-01", "2025-12-31"

    hourly = fetch_weather_open_meteo(lat, lon, start, end)
    print(">>> hourly keys:", hourly.keys())
    #print(">>> first 5 times:", list(hourly["time"][:5]))

    df = pd.DataFrame(hourly)
    records = []

    for i, ts in enumerate(df["time"]):
        row = df.iloc[i].to_dict()
        rec = generate_record(ts, row, lat, lon)
        records.append(rec)

    save_records(records)
    print(">>> records len:", len(records))
    print(">>> first record type:", type(records[0]) if records else None)
    print(">>> first record:", records[0] if records else None)

    print(">>> liczba rekordów do zapisu:", len(records))
    print(f"Zapisano {len(records)} rekordów do {PARQUET_FILE}")

if __name__ == "__main__":
    main()
