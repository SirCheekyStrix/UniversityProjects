"""
train.py — trenuje model na 5 latach danych dla wszystkich zmiennych.
Uruchamiaj raz w tygodniu. Czas: ~5-15 minut na GPU.

Użycie:
    python train.py
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from datetime import datetime

from weather_model import (
    fetch_training_data, add_features,
    WeatherTCN, FEATURES_IN, TARGETS, INPUT_HOURS, OUTPUT_HOURS,
    BATCH_SIZE, EPOCHS, LR,
    MODEL_PATH, SCALER_IN_PATH, SCALER_T_PATH, MODEL_DIR
)


class SeqDataset(torch.utils.data.IterableDataset):
    """Generuje sekwencje treningowe w locie — nie trzyma wszystkich w RAM."""
    def __init__(self, X, Y, shuffle=False):
        self.X       = X
        self.Y       = Y
        self.shuffle = shuffle
        self.n       = max(0, len(X) - INPUT_HOURS - OUTPUT_HOURS + 1)

    def __len__(self):
        return self.n

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in indices:
            x = torch.tensor(self.X[i : i + INPUT_HOURS].T,                           dtype=torch.float32)
            y = torch.tensor(self.Y[i + INPUT_HOURS : i + INPUT_HOURS + OUTPUT_HOURS].flatten(), dtype=torch.float32)
            yield x, y



def train():
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] === TRENING MODELU ===")
    print(f"Zmienne wyjściowe: {TARGETS}")

    # Usuń stare pliki
    for f in [MODEL_PATH, SCALER_IN_PATH, SCALER_T_PATH]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Usunięto: {f}")

    # Pobierz dane
    print("\nPobieram dane treningowe (5 lat)...")
    df = fetch_training_data(years_back=5)
    df = add_features(df)
    print(f"Zbiór: {len(df)} godzin | cech wejściowych: {len(FEATURES_IN)}")

    # ── Skalowanie wejść ──────────────────────────────────────────
    scaler_in = MinMaxScaler()
    X_scaled  = scaler_in.fit_transform(df[FEATURES_IN])
    X_scaled  = X_scaled.astype(np.float32)
    joblib.dump(scaler_in, SCALER_IN_PATH)

    # ── Skalowanie wyjść — osobny scaler dla każdej zmiennej ──────
    scalers_out = {}
    Y_cols      = []
    for name in TARGETS:
        sc  = MinMaxScaler(feature_range=(0.05, 0.95))  # unika zer/jedynek → stabilniejsze gradienty
        col = sc.fit_transform(df[[name]]).flatten()
        scalers_out[name] = sc
        Y_cols.append(col)
    joblib.dump(scalers_out, SCALER_T_PATH)

    # Y_matrix: [T, n_targets]
    # Y_matrix: [T, n_targets]
    Y_matrix = np.column_stack(Y_cols)
    Y_matrix  = Y_matrix.astype(np.float32)

    # Podział i DataLoader — sekwencje generowane w locie (brak kopii w RAM)
    split  = int(len(X_scaled) * 0.85)
    tr_ds  = SeqDataset(X_scaled[:split],  Y_matrix[:split],  shuffle=True)
    v_ds   = SeqDataset(X_scaled[split:],  Y_matrix[split:],  shuffle=False)
    tr_dl  = DataLoader(tr_ds, batch_size=BATCH_SIZE, num_workers=0)
    v_dl   = DataLoader(v_ds,  batch_size=BATCH_SIZE, num_workers=0)
    print(f'  Sekwencje train: {len(tr_ds)} | val: {len(v_ds)}')

    # ── Optymalizacje CPU (Xeon Platinum) ────────────────────────
    import multiprocessing
    if not torch.cuda.is_available():
        n_cores = multiprocessing.cpu_count()
        torch.set_num_threads(max(1, n_cores - 2))
        torch.set_num_interop_threads(max(1, n_cores // 4))
        print(f"CPU mode: {n_cores} rdzeni, {torch.get_num_threads()} wątków PyTorch")

    # ── Model ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_out  = OUTPUT_HOURS * len(TARGETS)
    model  = WeatherTCN(len(FEATURES_IN), n_out).to(device)

    # Inicjalizacja wag — zapobiega NaN w ep.1 przy dużym wyjściu
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # LR zmniejszone 5x względem poprzedniego — duże wyjście wymaga ostrożności
    effective_lr = LR / 5
    opt    = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=1e-4)
    crit   = nn.HuberLoss(delta=0.1)
    sched  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=200, T_mult=2, eta_min=1e-7
    )

    best, patience_cnt, PATIENCE = float("inf"), 0, 100
    print(f"\nTrening na {device} | wyjść={n_out} | batch={BATCH_SIZE}")

    for ep in range(EPOCHS):
        model.train()
        tl = 0.0
        n_tr_batches = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            n_tr_batches += 1
        sched.step()

        model.eval()
        vl = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, yb in v_dl:
                vl += crit(model(xb.to(device)), yb.to(device)).item()
                n_val_batches += 1

        avg_t = tl / max(n_tr_batches, 1)
        avg_v = vl / max(n_val_batches, 1)

        if np.isnan(avg_t) or np.isnan(avg_v):
            print(f"NaN w epoce {ep+1}! Sprawdź dane wejściowe lub zmniejsz LR.")
            # Spróbuj załadować ostatni dobry checkpoint
            if os.path.exists(MODEL_PATH) and best < float("inf"):
                print("  Wczytuję ostatni dobry checkpoint...")
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            break

        if (ep + 1) % 50 == 0:
            print(f"  Ep {ep+1:04d} | Train {avg_t:.5f} | Val {avg_v:.5f}")

        if avg_v < best:
            best = avg_v
            torch.save(model.state_dict(), MODEL_PATH)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop @ ep {ep+1}")
                break

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    t_range = scalers_out["temperature"].data_range_[0]
    rmse_c  = np.sqrt(best) * t_range
    print(f"\nNajlepszy val loss: {best:.5f} | RMSE temp: ~{rmse_c:.2f}°C")
    print(f"Model zapisany: {MODEL_PATH}")

    # Metadane
    meta = {
        "trained_at":    datetime.now().isoformat(),
        "val_loss":      float(best),
        "rmse_celsius":  float(rmse_c),
        "samples_train": len(tr_ds),
        "samples_val":   len(v_ds),
        "features_in":   FEATURES_IN,
        "targets":       TARGETS,
    }
    meta_path = os.path.join(MODEL_DIR, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadane: {meta_path}")


if __name__ == "__main__":
    train()