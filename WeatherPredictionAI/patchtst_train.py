"""
patchtst_train.py — trenuje model PatchTST do prognozy pogody 24h.

Użycie:
    python patchtst_train.py
    python patchtst_train.py --years 5 --epochs 500 --batch 8
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import joblib
from sklearn.preprocessing import MinMaxScaler

from model_swap import TrainingContext
from patchtst_model import (
    LAT, LON, INPUT_HOURS, OUTPUT_HOURS, PATCH_LENGTH, PATCH_STRIDE,
    FEATURES_IN, TARGETS,
    MODEL_DIR, MODEL_PATH, SCALER_IN_PATH, SCALER_T_PATH,
    build_model, fetch_training_data, add_features,
    BATCH_SIZE, EPOCHS, LR,
)

# =============================
# DATASET
# =============================

class SeqDataset(IterableDataset):
    """Generuje sekwencje (X, Y) bez kopiowania całej tablicy."""
    def __init__(self, X, Y, shuffle=False):
        self.X       = X
        self.Y       = Y
        self.n       = len(X) - INPUT_HOURS - OUTPUT_HOURS + 1
        self.shuffle = shuffle

    def __len__(self):
        return self.n

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in indices:
            x = torch.tensor(self.X[i : i + INPUT_HOURS], dtype=torch.float32)
            y = torch.tensor(self.Y[i + INPUT_HOURS : i + INPUT_HOURS + OUTPUT_HOURS], dtype=torch.float32)
            yield x, y   # x: [INPUT_HOURS, n_features],  y: [OUTPUT_HOURS, n_targets]


# =============================
# TRENING
# =============================

def train(years_back=5, epochs=EPOCHS, batch_size=BATCH_SIZE):
    print(f"[{pd.Timestamp.now():%Y-%m-%d %H:%M}] === TRENING PatchTST ===")
    print(f"Cechy wejściowe: {len(FEATURES_IN)}  |  Wyjścia: {len(TARGETS)}  |  Input: {INPUT_HOURS}h  |  Output: {OUTPUT_HOURS}h")

    # TrainingContext — zapisuje do tmp, podmienia atomowo po sukcesie
    global MODEL_PATH, SCALER_IN_PATH, SCALER_T_PATH
    _ctx = TrainingContext(MODEL_DIR)
    _ctx.__enter__()
    tmp = _ctx.tmp_dir
    # Nadpisz ścieżki zapisu na tmp
    import patchtst_model as _pm
    _orig_model_dir = _pm.MODEL_DIR
    _pm.MODEL_DIR      = tmp
    _pm.MODEL_PATH     = os.path.join(tmp, os.path.basename(MODEL_PATH))
    _pm.SCALER_IN_PATH = os.path.join(tmp, os.path.basename(SCALER_IN_PATH))
    _pm.SCALER_T_PATH  = os.path.join(tmp, os.path.basename(SCALER_T_PATH))
    MODEL_PATH     = _pm.MODEL_PATH
    SCALER_IN_PATH = _pm.SCALER_IN_PATH
    SCALER_T_PATH  = _pm.SCALER_T_PATH

    # ── Dane ──────────────────────────────────────────────────────────────
    print("Pobieram dane treningowe...")
    df = fetch_training_data(years_back=years_back)
    df = add_features(df)

    # Upewnij się że wszystkie cechy są dostępne
    missing = [f for f in FEATURES_IN + TARGETS if f not in df.columns]
    if missing:
        print(f"BŁĄD: brakujące kolumny: {missing}")
        sys.exit(1)

    print(f"Zbiór: {len(df)} godzin")

    # ── Skalowanie ────────────────────────────────────────────────────────
    scaler_in = MinMaxScaler()
    X_scaled  = scaler_in.fit_transform(df[FEATURES_IN].astype(np.float32)).astype(np.float32)
    joblib.dump(scaler_in, SCALER_IN_PATH)

    scalers_out = {}
    Y_cols      = []
    for name in TARGETS:
        sc   = MinMaxScaler()
        col  = sc.fit_transform(df[[name]].astype(np.float32)).astype(np.float32).flatten()
        scalers_out[name] = sc
        Y_cols.append(col)
    joblib.dump(scalers_out, SCALER_T_PATH)

    Y_matrix = np.column_stack(Y_cols).astype(np.float32)  # [N, n_targets]

    # ── Train/Val split ───────────────────────────────────────────────────
    split    = int(len(X_scaled) * 0.85)
    train_ds = SeqDataset(X_scaled[:split],  Y_matrix[:split],  shuffle=True)
    val_ds   = SeqDataset(X_scaled[split:],  Y_matrix[split:],  shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=0)

    print(f"  Train: ~{len(train_ds)} sekwencji  |  Val: ~{len(val_ds)} sekwencji")

    # ── Model ─────────────────────────────────────────────────────────────
    n_cores = os.cpu_count() or 1
    torch.set_num_threads(max(1, n_cores - 1))
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CPU mode: {n_cores} rdzeni  |  device: {device}")

    model     = build_model(len(FEATURES_IN)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)  # mniejszy weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=LR/50)
    criterion = nn.HuberLoss(delta=0.5)

    # Indeksy TARGETS w FEATURES_IN (PatchTST przewiduje wszystkie kanały)
    target_indices = [FEATURES_IN.index(t) for t in TARGETS]

    # ── Pętla treningowa ──────────────────────────────────────────────────
    best_val   = float("inf")
    patience   = 100  # więcej cierpliwości — model może mieć plateau przed poprawą
    no_improve = 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train    = 0

        for X_batch, Y_batch in train_loader:
            # X_batch: [B, INPUT_HOURS, n_features]
            # Y_batch: [B, OUTPUT_HOURS, n_targets]
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            out  = model(past_values=X_batch)
            pred = out.prediction_outputs   # [B, OUTPUT_HOURS, n_features]

            # Porównaj tylko kanały odpowiadające TARGETS
            pred_targets = pred[:, :, target_indices]   # [B, 24, 9]
            loss = criterion(pred_targets, Y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            n_train    += X_batch.size(0)

        scheduler.step()

        # Walidacja co 10 epok
        if ep % 10 == 0 or ep == 1:
            model.eval()
            val_loss = 0.0
            n_val    = 0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)
                    out     = model(past_values=X_batch)
                    pred    = out.prediction_outputs[:, :, target_indices]
                    val_loss += criterion(pred, Y_batch).item() * X_batch.size(0)
                    n_val    += X_batch.size(0)

            tl = train_loss / n_train
            vl = val_loss   / n_val
            print(f"  Ep {ep:04d} | Train {tl:.5f} | Val {vl:.5f}", flush=True)

            if vl < best_val:
                best_val   = vl
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 10
                if no_improve >= patience:
                    print(f"Early stop @ ep {ep}  |  best val: {best_val:.5f}")
                    break

    # Zapisz najlepszy model
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)

    # RMSE temperatury (orientacyjne)
    model.eval()
    rmse_sum, rmse_n = 0.0, 0
    sc_temp = scalers_out["temperature"]
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device)
            out     = model(past_values=X_batch)
            pred    = out.prediction_outputs[:, :, target_indices].cpu().numpy()
            true    = Y_batch.numpy()
            # Tylko temperatura (index 0)
            p_temp = sc_temp.inverse_transform(pred[:, :, 0].reshape(-1, 1)).flatten()
            t_temp = sc_temp.inverse_transform(true[:, :, 0].reshape(-1, 1)).flatten()
            rmse_sum += np.sum((p_temp - t_temp) ** 2)
            rmse_n   += len(p_temp)

    rmse = np.sqrt(rmse_sum / rmse_n) if rmse_n > 0 else float("nan")
    print(f"\nModel zapisany: {MODEL_PATH}")
    print(f"Najlepszy val loss: {best_val:.5f} | RMSE temp: ~{rmse:.2f}°C")
    # Atomowa podmiana
    _pm.MODEL_DIR = _orig_model_dir
    _ctx.__exit__(None, None, None)
    print("Model podmieniony atomowo — API moze korzystac z nowego modelu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",  type=int, default=5)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(years_back=args.years, epochs=args.epochs, batch_size=args.batch)