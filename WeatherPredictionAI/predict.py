"""
predict.py — pobiera najnowsze dane i generuje prognozę 24h.
Zapisuje wynik do JSON + opcjonalnie wielopanelowy wykres PNG.

Użycie:
    python predict.py              # tylko JSON
    python predict.py --plot       # JSON + wykres PNG
    python predict.py --print      # JSON + wydruk w konsoli
"""
import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from weather_model import (
    fetch_recent_window, add_features, predict, TARGETS,
    FORECAST_PATH, MODEL_DIR, INPUT_HOURS, OUTPUT_HOURS
)

LABELS = {
    "temperature":  ("Temperatura",            "°C",   "crimson"),
    "apparent_temp":("Odczuwalna temp.",        "°C",   "salmon"),
    "wind_speed":   ("Wiatr",                  "km/h", "steelblue"),
    "wind_gust":    ("Porywy",                 "km/h", "darkorange"),
    "pressure":     ("Ciśnienie",              "hPa",  "seagreen"),
    "rain_rate":    ("Opady",                  "mm/h", "royalblue"),
    "humidity":     ("Wilgotność",             "%",    "teal"),
    "dewpoint":     ("Punkt rosy",             "°C",   "mediumturquoise"),
    "cloudcover":   ("Zachmurzenie",           "%",    "slategray"),
    "uv_index":     ("UV Index",               "",     "goldenrod"),
    "pop_pct":      ("Prawdop. opadów (PoP)",  "%",    "mediumpurple"),
}


def run_prediction(save_plot=False, print_output=False, df_input=None):
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Predykcja...")

    if df_input is not None:
        df = add_features(df_input.copy())
    else:
        try:
            df = fetch_recent_window()
        except Exception as e:
            print(f"BŁĄD pobierania danych: {e}", file=sys.stderr)
            sys.exit(1)
        df = add_features(df)

    if len(df) < INPUT_HOURS:
        print(f"BŁĄD: za mało danych ({len(df)} < {INPUT_HOURS})", file=sys.stderr)
        sys.exit(1)

    try:
        forecast_df = predict(df)
    except FileNotFoundError as e:
        print(f"BŁĄD: {e}", file=sys.stderr)
        sys.exit(1)

    # Dodaj prawdopodobieństwo opadów (PoP = C × A)
    # C = pewność że deszcz wystąpi — funkcja rain_rate
    # A = pokrycie obszaru — funkcja wilgotności i ciśnienia
    def calc_pop(rain_rate, humidity, pressure):
        # C: rośnie eksponencjalnie z rain_rate
        # przy 0.1mm/h → ~28%, przy 0.5mm/h → ~81%, przy 1mm/h → ~96%
        C = 1.0 - np.exp(-rain_rate / 0.3)
        # A: wilgotność bazowa + korekta ciśnieniem
        # niskie ciśnienie (<1005 hPa) = front = większy obszar opadów
        pressure_factor = np.clip((1020 - pressure) / 30, 0, 1)  # 0 przy 1020hPa, 1 przy 990hPa
        A = (humidity / 100) * (0.6 + 0.4 * pressure_factor)
        pop = np.clip(C * A * 100, 0, 100)
        return round(float(pop), 1)

    # Pobierz wilgotność i ciśnienie z ostatnich danych (brak w forecast_df)
    # Użyj historycznych wartości jako przybliżenia trendu
    hist_humidity = df["humidity"].values[-OUTPUT_HOURS:]
    hist_pressure = df["pressure"].values[-OUTPUT_HOURS:]

    # Zbuduj JSON
    forecast_rows = []
    for i, row in enumerate(forecast_df.itertuples()):
        entry = {"timestamp": str(row.timestamp)}
        for col in TARGETS:
            val = getattr(row, col)
            entry[col] = round(float(val), 1)
        # PoP
        hum  = float(hist_humidity[i]) if i < len(hist_humidity) else float(hist_humidity[-1])
        pres = float(row.pressure)
        entry["pop_pct"]  = calc_pop(float(row.rain_rate), hum, pres)
        entry["uv_index"] = round(float(row.uv_index), 1)
        forecast_rows.append(entry)

    pops = [r["pop_pct"] for r in forecast_rows]
    uvs  = [r["uv_index"] for r in forecast_rows]
    result = {
        "generated_at": datetime.now().isoformat(),
        "forecast": forecast_rows,
        "summary": {
            **{
                col: {
                    "min":  round(float(forecast_df[col].min()), 1),
                    "max":  round(float(forecast_df[col].max()), 1),
                    "mean": round(float(forecast_df[col].mean()), 1),
                }
                for col in TARGETS
            },
            "uv_index": {
                "max":  round(max(uvs), 1),
                "mean": round(sum(uvs) / len(uvs), 1),
            },
            "pop_pct": {
                "max":  round(max(pops), 1),
                "mean": round(sum(pops) / len(pops), 1),
                "hours_above_50pct": sum(1 for p in pops if p >= 50),
            }
        }
    }

    with open(FORECAST_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Prognoza zapisana: {FORECAST_PATH}")

    # Zapis do InfluxDB
    try:
        from influx_writer import write_weather_24h
        n = write_weather_24h(result)
        print(f"  → InfluxDB predictions: {n} punktów")
    except Exception as e:
        print(f"  ⚠ InfluxDB pominięty: {e}")
    t  = result["summary"]["temperature"]
    at = result["summary"]["apparent_temp"]
    w  = result["summary"]["wind_speed"]
    g  = result["summary"]["wind_gust"]
    r  = result["summary"]["rain_rate"]
    p  = result["summary"]["pressure"]
    h  = result["summary"]["humidity"]
    c  = result["summary"]["cloudcover"]
    uv = result["summary"]["uv_index"]
    print(f"  Temp:       min {t['min']}°C / max {t['max']}°C / śr {t['mean']}°C")
    print(f"  Odczuwalna: min {at['min']}°C / max {at['max']}°C")
    print(f"  Wiatr:      max {w['max']} km/h | Porywy: max {g['max']} km/h")
    print(f"  Ciśnienie śr: {p['mean']} hPa")
    print(f"  Wilgotność: śr {h['mean']}% | Zachmurzenie: śr {c['mean']}%")
    print(f"  UV max: {uv['max']} | Opady max: {r['max']} mm/h")
    pop = result["summary"]["pop_pct"]
    print(f"  PoP max: {pop['max']}% | śr: {pop['mean']}% | "
          f"godzin ≥50%: {pop['hours_above_50pct']}")

    if print_output:
        header = (f"{'Godz':<20} {'Temp':>6} {'Odcz':>6} {'Wiatr':>7} {'Porywy':>8} "
                  f"{'Ciśn':>8} {'Opady':>7} {'RH':>5} {'Rosy':>6} {'Chmury':>7} {'UV':>4}")
        print(f"\n{header}")
        print("-" * len(header))
        for row in forecast_rows:
            print(f"  {row['timestamp'][:16]}  "
                  f"{row['temperature']:>+5.1f}°C "
                  f"{row['apparent_temp']:>+5.1f}°C "
                  f"{row['wind_speed']:>5.1f}km/h "
                  f"{row['wind_gust']:>6.1f}km/h "
                  f"{row['pressure']:>7.1f}hPa "
                  f"{row['rain_rate']:>5.1f}mm/h "
                  f"{row['humidity']:>4.0f}%RH "
                  f"{row['dewpoint']:>+5.1f}°C "
                  f"{row['cloudcover']:>5.0f}% "
                  f"UV:{row['uv_index']:>3.1f} "
                  f"PoP:{row['pop_pct']:>4.0f}%")

    if save_plot:
        _save_plot(df, forecast_df, forecast_rows)

    return result


def _save_plot(df, forecast_df, forecast_rows):
    import matplotlib.dates as mdates

    now_ts    = pd.Timestamp(datetime.now()).floor("h")
    history_hours = 48

    # Historia: ostatnie 48h rzeczywistych obserwacji (tylko przeszłość)
    hist_df   = df[df["timestamp"] <= now_ts].tail(history_hours)
    hist_ts   = hist_df["timestamp"].values
    pred_ts   = forecast_df["timestamp"].values

    plot_cols = list(TARGETS) + ["pop_pct"]
    fig, axes = plt.subplots(len(plot_cols), 1,
                             figsize=(14, 3.2 * len(plot_cols)), sharex=False)
    fig.suptitle(f"Prognoza pogody — TCN  ({datetime.now():%Y-%m-%d %H:%M} UTC)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, col in zip(axes, plot_cols):
        label, unit, color = LABELS[col]

        # Dane historyczne
        if col == "pop_pct":
            hist_vals = np.zeros(len(hist_ts))
        else:
            hist_vals = hist_df[col].values

        # Dane prognozy
        if col == "pop_pct":
            pred_vals = np.array([r["pop_pct"] for r in forecast_rows])
        else:
            pred_vals = forecast_df[col].values

        # Linia łącząca: ostatni punkt historii → pierwsza godzina prognozy
        last_val  = hist_vals[-1] if len(hist_vals) > 0 else pred_vals[0]
        join_ts   = np.concatenate([[hist_ts[-1]], pred_ts])
        join_vals = np.concatenate([[last_val],    pred_vals])

        # Rysuj
        ax.plot(hist_ts,  hist_vals,  color="gray",  linewidth=2,
                label="Historia (48h)", alpha=0.9, zorder=2)
        ax.plot(join_ts,  join_vals,  color=color,   linewidth=2,
                linestyle="--", marker="o", markersize=3,
                label="Prognoza 24h", zorder=3)
        ax.fill_between(join_ts, join_vals, alpha=0.07, color=color)

        # Linia "Teraz"
        ax.axvline(x=now_ts, color="black", linestyle=":",
                   linewidth=1.2, label=f"Teraz ({now_ts:%H:%M})", zorder=4)

        # Próg 50% dla PoP
        if col == "pop_pct":
            ax.axhline(y=50, color="red", linestyle="--",
                       linewidth=0.8, alpha=0.6, label="50% próg")
            ax.set_ylim(0, 105)

        # Adnotacje min/max na prognozie
        if len(pred_vals) > 0:
            peak_i   = int(np.argmax(pred_vals))
            trough_i = int(np.argmin(pred_vals))
            ax.annotate(f"{pred_vals[peak_i]:.1f}{unit}",
                        xy=(pred_ts[peak_i], pred_vals[peak_i]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=color, fontweight="bold")
            if peak_i != trough_i:
                ax.annotate(f"{pred_vals[trough_i]:.1f}{unit}",
                            xy=(pred_ts[trough_i], pred_vals[trough_i]),
                            xytext=(0, -14), textcoords="offset points",
                            ha="center", fontsize=8, color=color)

        ax.set_ylabel(f"{label} [{unit}]", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        # Ogranicz oś X: tylko 48h historii + 24h prognozy
        ax.set_xlim(pd.Timestamp(hist_ts[0]), pd.Timestamp(pred_ts[-1]))

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "latest_forecast.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wykres zapisany: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot",  action="store_true", help="Zapisz wykres PNG")
    parser.add_argument("--print", action="store_true", dest="print_out",
                        help="Wydrukuj tabelę godzinową")
    args = parser.parse_args()
    run_prediction(save_plot=args.plot, print_output=args.print_out)