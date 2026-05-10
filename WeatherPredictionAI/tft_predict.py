"""
tft_predict.py — generuje 10-dniową prognozę pogody z modelu TFT.

Użycie:
    python tft_predict.py
    python tft_predict.py --print
    python tft_predict.py --plot

Wynik:
    tft_model_files/tft_forecast.json  — prognoza z kwantylami
    tft_model_files/tft_forecast.png   — wykres (z --plot)
"""
import argparse
import json
import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
# PyTorch 2.6: weights_only=True domyślnie — wyłącz dla checkpointów pytorch_forecasting
_torch_load_orig = torch.load
torch.load = lambda *a, **kw: _torch_load_orig(*a, **{**kw, "weights_only": False})
from datetime import datetime

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from tft_model import (
    MODEL_DIR, FORECAST_PATH,
    TARGETS, KNOWN_FUTURE_REALS, KNOWN_FUTURE_CATS, OBSERVED_PAST,
    ENCODER_LEN, PRED_LEN, LOCATION_ID,
    fetch_recent_data, add_features, compute_pop
)

UNITS = {
    "temp_max":        "°C",
    "temp_min":        "°C",
    "precip_sum":      "mm",
    "wind_max":        "km/h",
    "wind_mean":       "km/h",
    "pressure_mean":   "hPa",
    "cloudcover_mean": "%",
    "humidity_mean":   "%",
}

LABELS = {
    "temp_max":        "Temp max",
    "temp_min":        "Temp min",
    "precip_sum":      "Opady",
    "wind_max":        "Wiatr maks",
    "wind_mean":       "Wiatr śr",
    "pressure_mean":   "Ciśnienie",
    "cloudcover_mean": "Zachmurzenie",
    "humidity_mean":   "Wilgotność",
}


def find_checkpoint() -> str:
    """Znajdź najnowszy checkpoint TFT."""
    ckpts = glob.glob(os.path.join(MODEL_DIR, "tft_weather*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"Brak checkpointu w {MODEL_DIR}.\n"
            "Uruchom najpierw: python tft_train.py"
        )
    return max(ckpts, key=os.path.getmtime)


def predict_10days(df_input=None) -> dict:
    """Generuje 10-dniową prognozę z kwantylami [0.1, 0.5, 0.9]."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Pobieranie danych...")
    if df_input is not None:
        df = add_features(df_input.copy())
    else:
        raw = fetch_recent_data()   # 65 dni historii + 16 dni NWP
        df  = add_features(raw)

    today = pd.Timestamp.now().normalize()

    # ── Wszystkie kolumny wymagane przez model ──────────────────────────────
    all_num_cols = (
        TARGETS
        + KNOWN_FUTURE_REALS
        + OBSERVED_PAST
        + ["temp_mean", "d_pressure_3d", "precip_7d_sum", "temp_anomaly", "nwp_error_temp"]
    )
    # Usuń duplikaty zachowując kolejność
    seen = set()
    all_num_cols = [c for c in all_num_cols if not (c in seen or seen.add(c))]

    # ── Wypełnij WSZYSTKIE NaN przed zbudowaniem datasetu ──────────────────
    for col in all_num_cols:
        if col in df.columns:
            # ffill: przenieś ostatnią obserwację w przyszłość
            df[col] = df[col].ffill()
            # bfill: jeśli na początku są NaN
            df[col] = df[col].bfill()
            # Ostateczna bezpieczna wartość = 0 (nie powinno wystąpić)
            df[col] = df[col].fillna(0.0)

    # Upewnij się że NWP dla przyszłych dni jest wypełnione
    nwp_cols = [c for c in KNOWN_FUTURE_REALS if c.startswith("nwp_")]
    target_map = {
        "nwp_temp_max":   "temp_max",
        "nwp_temp_min":   "temp_min",
        "nwp_precip":     "precip_sum",
        "nwp_pressure":   "pressure_mean",
        "nwp_wind_max":   "wind_max",
        "nwp_cloudcover": "cloudcover_mean",
    }
    for nwp_col, obs_col in target_map.items():
        if nwp_col in df.columns and obs_col in df.columns:
            mask = df[nwp_col].isna() | (df[nwp_col] == 0)
            df.loc[mask, nwp_col] = df.loc[mask, obs_col]
            df[nwp_col] = df[nwp_col].ffill().bfill().fillna(0.0)

    # Fizyczne ograniczenia
    df["precip_sum"]      = df["precip_sum"].clip(lower=0)
    df["nwp_precip"]      = df["nwp_precip"].clip(lower=0)
    df["cloudcover_mean"] = df["cloudcover_mean"].clip(0, 100)
    df["nwp_cloudcover"]  = df["nwp_cloudcover"].clip(0, 100)
    df["humidity_mean"]   = df["humidity_mean"].clip(0, 100)

    # ── Zbuduj okno predykcji: ENCODER_LEN historii + PRED_LEN przyszłości ─
    df_hist   = df[df["date"] <= today].tail(ENCODER_LEN)
    df_future = df[df["date"] >  today].head(PRED_LEN)

    if len(df_hist) < ENCODER_LEN // 2:
        raise ValueError(f"Za mało historii: {len(df_hist)} dni (potrzeba {ENCODER_LEN//2})")
    if len(df_future) < PRED_LEN:
        raise ValueError(
            f"Za mało danych przyszłości z API: {len(df_future)} dni (potrzeba {PRED_LEN})."
            "Open-Meteo Forecast API powinno zwracać 16 dni — sprawdź połączenie."
        )

    df_full = pd.concat([df_hist, df_future], ignore_index=True)

    # Przereset time_idx od 0
    df_full["time_idx"] = range(len(df_full))
    df_full["group_id"] = LOCATION_ID

    future_dates = df_future["date"].values

    # Załaduj model
    print("Ładowanie modelu TFT...")
    ckpt = find_checkpoint()
    tft  = TemporalFusionTransformer.load_from_checkpoint(ckpt)
    tft.eval()

    # Dataset do predykcji
    predict_ds = TimeSeriesDataSet(
        df_full,
        time_idx             = "time_idx",
        target               = TARGETS,
        group_ids            = ["group_id"],
        min_encoder_length   = ENCODER_LEN // 2,
        max_encoder_length   = ENCODER_LEN,
        min_prediction_length= PRED_LEN,
        max_prediction_length= PRED_LEN,
        time_varying_known_reals        = ["time_idx"] + KNOWN_FUTURE_REALS,
        time_varying_known_categoricals = KNOWN_FUTURE_CATS,
        time_varying_unknown_reals      = TARGETS + OBSERVED_PAST,
        allow_missing_timesteps         = True,
        predict_mode                    = True,
    )
    loader = predict_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

    # Predykcja z kwantylami
    print("Generowanie prognozy...")
    with torch.no_grad():
        raw_preds = tft.predict(
            loader,
            mode               = "quantiles",
            return_index       = True,
            trainer_kwargs     = {"accelerator": "cpu"},
        )

    # ── NWP z API jako punkt odniesienia ──────────────────────────────────────
    # Mapowanie target → kolumna NWP w df_future
    NWP_COL = {
        "temp_max":        "nwp_temp_max",
        "temp_min":        "nwp_temp_min",
        "precip_sum":      "nwp_precip",
        "wind_max":        "nwp_wind_max",
        "pressure_mean":   "nwp_pressure",
        "cloudcover_mean": "nwp_cloudcover",
    }
    # Wagi blendu: ile % pochodzi z NWP (reszta z TFT)
    # Im bliżej, tym bardziej ufamy NWP (szczególnie dla temp i opadów)
    def nwp_weight(day_i: int, target: str) -> float:
        if target == "precip_sum":
            weights = [0.95, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]
        elif "temp" in target:
            # TFT ma bias zimna — silnie ufaj NWP dopóki model nie zostanie retrenowany
            weights = [0.92, 0.90, 0.87, 0.83, 0.78, 0.72, 0.65, 0.58, 0.50, 0.42]
        elif target == "wind_max":
            weights = [0.85, 0.82, 0.78, 0.73, 0.67, 0.60, 0.53, 0.46, 0.39, 0.32]
        elif target == "pressure_mean":
            weights = [0.88, 0.85, 0.81, 0.76, 0.70, 0.63, 0.56, 0.49, 0.42, 0.35]
        else:
            weights = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
        return weights[min(day_i, len(weights)-1)]

    # Wyciągnij predykcje TFT dla wszystkich dni i targetów
    tft_preds = {}  # target → list of (q10, q50, q90)
    for t_i, target in enumerate(TARGETS):
        target_vals = []
        for day_i in range(len(future_dates)):
            try:
                out = raw_preds.output
                if isinstance(out, list):
                    vals = out[t_i][0, day_i, :].numpy()
                elif out.ndim == 4:
                    vals = out[0, day_i, t_i, :].numpy()
                else:
                    vals = out[0, day_i, :].numpy()
                target_vals.append((float(vals[0]), float(vals[1]), float(vals[2])))
            except Exception:
                target_vals.append((np.nan, np.nan, np.nan))
        tft_preds[target] = target_vals

    # ── Zbuduj forecast z blendingiem TFT + NWP ───────────────────────────
    forecast_days = []
    for day_i, fdate in enumerate(future_dates):
        day_entry = {
            "date": str(pd.Timestamp(fdate).date()),
            "day":  int(day_i + 1),
        }
        nwp_row = df_future.iloc[day_i]

        for target in TARGETS:
            q10_tft, q50_tft, q90_tft = tft_preds[target][day_i]

            # Pobierz wartość NWP dla tego targetu
            nwp_col = NWP_COL.get(target)
            if nwp_col and nwp_col in nwp_row.index:
                nwp_val = float(nwp_row[nwp_col])
            else:
                nwp_val = q50_tft  # brak NWP → użyj TFT

            w = nwp_weight(day_i, target)  # waga NWP

            # Blend: q50 = w*NWP + (1-w)*TFT
            # Dla q10/q90 zachowaj rozpiętość z TFT (niepewność) wokół nowej mediany
            spread_lo = q50_tft - q10_tft
            spread_hi = q90_tft - q50_tft

            if np.isnan(q50_tft):
                q50 = nwp_val
                q10 = nwp_val - 1.0
                q90 = nwp_val + 1.0
            else:
                q50 = w * nwp_val + (1 - w) * q50_tft
                q10 = q50 - abs(spread_lo)
                q90 = q50 + abs(spread_hi)

            # Fizyczne ograniczenia
            if target == "precip_sum":
                q10 = max(0.0, q10)
                q50 = max(0.0, q50)
                q90 = max(0.0, q90)
            elif target in ("wind_max", "wind_mean"):
                q10, q50, q90 = max(0.0, q10), max(0.0, q50), max(0.0, q90)
            elif target in ("cloudcover_mean", "humidity_mean"):
                q10 = max(0, min(100, q10))
                q50 = max(0, min(100, q50))
                q90 = max(0, min(100, q90))

            day_entry[target] = {
                "q10":    round(q10, 1),
                "median": round(q50, 1),
                "q90":    round(q90, 1),
                "nwp":    round(nwp_val, 1) if nwp_col else None,
                "tft":    round(q50_tft, 1) if not np.isnan(q50_tft) else None,
                "nwp_weight": round(w, 2),
            }

        # PoP z blended precip + NWP humidity/cloudcover
        day_entry["pop_pct"] = compute_pop({
            "precip_sum":      day_entry["precip_sum"]["median"],
            "humidity_mean":   day_entry["humidity_mean"]["median"],
            "cloudcover_mean": day_entry["cloudcover_mean"]["median"],
            "nwp_precip":      float(nwp_row.get("nwp_precip", 0) or 0),
        })

        forecast_days.append(day_entry)

    result = {
        "generated_at": datetime.now().isoformat(),
        "model":        "TFT",
        "horizon_days": PRED_LEN,
        "targets":      TARGETS,
        "quantiles":    [0.1, 0.5, 0.9],
        "forecast":     forecast_days,
    }

    with open(FORECAST_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"JSON zapisany: {FORECAST_PATH}")

    # Zapis do InfluxDB
    try:
        from influx_writer import write_weather_10d
        n = write_weather_10d(result)
        print(f"  → InfluxDB predictions: {n} punktów")
    except Exception as e:
        print(f"  ⚠ InfluxDB pominięty: {e}")

    return result


def print_forecast(result: dict):
    days = result["forecast"]
    header = (f"\n{'Data':<13}"
              f"{'Tmax':>7}{'Tmin':>7}"
              f"{'Opady':>8}{'PoP':>6}"
              f"{'Wiatr':>7}{'Ciśn':>8}"
              f"{'Chmury':>8}{'RH':>6}")
    print(header)
    print("─" * len(header))

    for d in days:
        tmax = d["temp_max"]["median"]
        tmin = d["temp_min"]["median"]
        prec = d["precip_sum"]["median"]
        pop  = d["pop_pct"]
        wind = d["wind_max"]["median"]
        pres = d["pressure_mean"]["median"]
        cld  = d["cloudcover_mean"]["median"]
        hum  = d["humidity_mean"]["median"]

        # Ikony pogody
        if pop >= 70:
            icon = "🌧"
        elif pop >= 40:
            icon = "🌦"
        elif cld > 70:
            icon = "☁"
        elif cld > 30:
            icon = "⛅"
        else:
            icon = "☀"

        print(f"  {d['date']} {icon}"
              f"  {tmax:>+5.1f}°C"
              f"  {tmin:>+5.1f}°C"
              f"  {prec:>5.1f}mm"
              f"  {pop:>4.0f}%"
              f"  {wind:>5.1f}km/h"
              f"  {pres:>7.1f}hPa"
              f"  {cld:>5.1f}%"
              f"  {hum:>4.1f}%")

        # Przedział niepewności + wartości NWP i TFT dla porównania
        tmax_lo = d["temp_max"]["q10"]
        tmax_hi = d["temp_max"]["q90"]
        tmin_lo = d["temp_min"]["q10"]
        tmin_hi = d["temp_min"]["q90"]
        nwp_tmax = d["temp_max"].get("nwp")
        nwp_tmin = d["temp_min"].get("nwp")
        nwp_rain = d["precip_sum"].get("nwp")
        w        = d["temp_max"].get("nwp_weight", 0)
        print(f"  {'':13}  "
              f"  [{tmax_lo:>+5.1f}÷{tmax_hi:>+5.1f}]"
              f"  [{tmin_lo:>+5.1f}÷{tmin_hi:>+5.1f}]"
              f"  NWP:{nwp_tmax:>+5.1f}/{nwp_tmin:>+5.1f}°C"
              f"  rain:{nwp_rain:>4.1f}mm"
              f"  (w_nwp={w:.0%})")


def save_plot(result: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib niedostępny — pomiń wykres")
        return

    days    = result["forecast"]
    dates   = pd.to_datetime([d["date"] for d in days])
    COLORS  = {
        "temp":     ("crimson",    "lightcoral"),
        "precip":   ("royalblue",  "lightblue"),
        "wind":     ("darkorange", "moccasin"),
        "pressure": ("seagreen",   "lightgreen"),
        "cloud":    ("slategray",  "lightgray"),
        "humidity": ("teal",       "paleturquoise"),
    }

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        f"Prognoza TFT 10 dni  —  {datetime.now():%Y-%m-%d %H:%M}",
        fontsize=14, fontweight="bold"
    )

    def get_vals(key):
        return (
            np.array([d[key]["q10"]    for d in days]),
            np.array([d[key]["median"] for d in days]),
            np.array([d[key]["q90"]    for d in days]),
        )

    # Panel 1: Temperatura
    ax = axes[0]
    lo_max, med_max, hi_max = get_vals("temp_max")
    lo_min, med_min, hi_min = get_vals("temp_min")
    ax.fill_between(dates, lo_max, hi_max, alpha=0.2, color="crimson", label="Zakres Tmax")
    ax.fill_between(dates, lo_min, hi_min, alpha=0.2, color="steelblue", label="Zakres Tmin")
    ax.plot(dates, med_max, "o-", color="crimson",   lw=2, label="Tmax")
    ax.plot(dates, med_min, "o-", color="steelblue", lw=2, label="Tmin")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    for i, (d, hi, lo) in enumerate(zip(dates, med_max, med_min)):
        ax.annotate(f"{hi:+.0f}", (d, hi), textcoords="offset points", xytext=(0, 6),
                    fontsize=7, ha="center", color="crimson")
        ax.annotate(f"{lo:+.0f}", (d, lo), textcoords="offset points", xytext=(0, -10),
                    fontsize=7, ha="center", color="steelblue")
    ax.set_ylabel("Temperatura [°C]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel 2: Opady + PoP
    ax = axes[1]
    lo_p, med_p, hi_p = get_vals("precip_sum")
    pops = np.array([d["pop_pct"] for d in days])
    ax2  = ax.twinx()
    ax.bar(dates, med_p, width=0.6, color="royalblue", alpha=0.7, label="Opady (mediana)")
    ax.fill_between(dates, lo_p, hi_p, alpha=0.2, color="royalblue")
    ax2.plot(dates, pops, "s--", color="navy", lw=1.5, ms=5, label="PoP %")
    ax2.axhline(50, color="red", lw=0.8, ls=":", alpha=0.5)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("PoP [%]", color="navy")
    ax.set_ylabel("Opady [mm]")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel 3: Wiatr
    ax = axes[2]
    lo_wmax, med_wmax, hi_wmax = get_vals("wind_max")
    lo_wmn,  med_wmn,  hi_wmn  = get_vals("wind_mean")
    ax.fill_between(dates, lo_wmax, hi_wmax, alpha=0.2, color="darkorange")
    ax.plot(dates, med_wmax, "o-", color="darkorange", lw=2, label="Wiatr max")
    ax.plot(dates, med_wmn,  "o--",color="goldenrod",  lw=1.5, label="Wiatr śr")
    ax.set_ylabel("Wiatr [km/h]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel 4: Ciśnienie
    ax = axes[3]
    lo_pr, med_pr, hi_pr = get_vals("pressure_mean")
    ax.fill_between(dates, lo_pr, hi_pr, alpha=0.15, color="seagreen")
    ax.plot(dates, med_pr, "o-", color="seagreen", lw=2, label="Ciśnienie śr")
    ax.set_ylabel("Ciśnienie [hPa]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel 5: Zachmurzenie + Wilgotność
    ax = axes[4]
    lo_cl, med_cl, hi_cl = get_vals("cloudcover_mean")
    lo_hu, med_hu, hi_hu = get_vals("humidity_mean")
    ax2 = ax.twinx()
    ax.fill_between(dates, lo_cl, hi_cl, alpha=0.2, color="slategray")
    ax.plot(dates, med_cl, "o-", color="slategray", lw=2, label="Zachmurzenie")
    ax2.fill_between(dates, lo_hu, hi_hu, alpha=0.1, color="teal")
    ax2.plot(dates, med_hu, "s--", color="teal", lw=1.5, ms=5, label="Wilgotność")
    ax.set_ylabel("Zachmurzenie [%]")
    ax2.set_ylabel("Wilgotność [%]", color="teal")
    ax.set_ylim(0, 110)
    ax2.set_ylim(0, 110)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Oś X
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%a"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "tft_forecast.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wykres zapisany: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", dest="print_out")
    parser.add_argument("--plot",  action="store_true")
    args = parser.parse_args()

    result = predict_10days()

    if args.print_out:
        print_forecast(result)
    else:
        # Zawsze krótkie podsumowanie
        print("\n10-dniowa prognoza (mediana):")
        for d in result["forecast"]:
            tmax = d["temp_max"]["median"]
            tmin = d["temp_min"]["median"]
            pop  = d["pop_pct"]
            icon = "🌧" if pop >= 70 else ("🌦" if pop >= 40 else "☀")
            print(f"  {d['date']}  {icon}  {tmax:>+5.1f}/{tmin:>+5.1f}°C  PoP:{pop:.0f}%")

    if args.plot:
        save_plot(result)