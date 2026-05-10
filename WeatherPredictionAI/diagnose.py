"""
diagnose.py — pokazuje co model widzi jako ostatnie dane wejściowe.
Uruchom żeby zrozumieć dlaczego prognoza jest niedoszacowana.
"""
import pandas as pd
import numpy as np
from weather_model import fetch_recent_window, add_features, FEATURES_IN, INPUT_HOURS

print("Pobieram ostatnie dane...")
df = fetch_recent_window()
df = add_features(df)

window = df[FEATURES_IN].iloc[-INPUT_HOURS:]
last   = df[FEATURES_IN].iloc[-1]
last24 = df[FEATURES_IN].iloc[-24:]

print(f"\n=== OSTATNIA GODZINA ({df['timestamp'].iloc[-1]}) ===")
for col in FEATURES_IN:
    print(f"  {col:<20} {last[col]:>8.2f}")

print(f"\n=== TEMP_850HPA — ostatnie 24h ===")
t850 = df["temp_850hpa"].iloc[-24:].values
ts   = df["timestamp"].iloc[-24:].values
for t, v in zip(ts, t850):
    print(f"  {str(t)[:16]}  {v:+.1f}°C")

print(f"\n=== ZMIANY CIŚNIENIA — ostatnia godzina ===")
print(f"  d_pressure_3h:   {last['d_pressure_3h']:+.2f} hPa")
print(f"  d_pressure_24h:  {last['d_pressure_24h']:+.2f} hPa")
print(f"  d_temp_6h:       {last['d_temp_6h']:+.2f}°C")

print(f"\n=== KIERUNEK WIATRU ===")
wd_sin = last["wind_dir_sin"]
wd_cos = last["wind_dir_cos"]
wd_deg = np.degrees(np.arctan2(wd_sin, wd_cos)) % 360
kierunki = {(337.5, 360): "N", (0, 22.5): "N", (22.5, 67.5): "NE",
            (67.5, 112.5): "E", (112.5, 157.5): "SE", (157.5, 202.5): "S",
            (202.5, 247.5): "SW", (247.5, 292.5): "W", (292.5, 337.5): "NW"}
kierunek = next((v for (lo, hi), v in kierunki.items() if lo <= wd_deg < hi), "?")
print(f"  Kierunek: {wd_deg:.0f}° ({kierunek})")
print(f"  SW/W = atlantyk (ciepło), NE/E = kontynent (zimno)")

print(f"\n=== TEMPERATURA POWIERZCHNIOWA — ostatnie 12h ===")
for _, row in df.tail(12).iterrows():
    print(f"  {str(row['timestamp'])[:16]}  {row['temperature']:+.1f}°C  "
          f"850hPa: {row['temp_850hpa']:+.1f}°C")