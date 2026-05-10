"""
tft_train.py — trenuje model TFT na danych dziennych z Open-Meteo.

Użycie:
    python tft_train.py
    python tft_train.py --years 7
    python tft_train.py --epochs 50 --batch 64

Wymagania:
    pip install pytorch-forecasting pytorch-lightning torch
"""
import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
_torch_load_orig = torch.load
torch.load = lambda *a, **kw: _torch_load_orig(*a, **{**kw, "weights_only": False})
from datetime import datetime

# pytorch-forecasting >= 1.0 używa lightning zamiast pytorch_lightning
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss

from model_swap import TrainingContext
from tft_model import (
    MODEL_DIR, MODEL_PATH, DATA_PATH,
    TARGETS, KNOWN_FUTURE_REALS, KNOWN_FUTURE_CATS, OBSERVED_PAST,
    ENCODER_LEN, PRED_LEN, LOCATION_ID,
    fetch_training_data, add_features
)

# ─────────────────────────────────────────────
# PARAMETRY
# ─────────────────────────────────────────────
DEFAULT_EPOCHS     = 150
DEFAULT_BATCH      = 64
DEFAULT_YEARS      = 5
LR                 = 1e-3
HIDDEN_SIZE        = 32     # mały model na ~1800 próbek — ogranicza overfitting
ATTENTION_HEADS    = 2
DROPOUT            = 0.3
HIDDEN_CONT_SIZE   = 16


def build_dataset(df: pd.DataFrame, is_train: bool = True) -> TimeSeriesDataSet:
    """Buduje TimeSeriesDataSet dla TFT."""

    max_idx   = df["time_idx"].max()
    # Walidacja na ostatnich 90 dniach (nie tylko PRED_LEN)
    val_days  = 90
    cutoff    = max_idx - val_days

    dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= cutoff] if is_train else df[df["time_idx"] > cutoff - ENCODER_LEN],
        time_idx             = "time_idx",
        target               = TARGETS,          # multi-target
        group_ids            = ["group_id"],
        min_encoder_length   = ENCODER_LEN // 2,
        max_encoder_length   = ENCODER_LEN,
        min_prediction_length= PRED_LEN,
        max_prediction_length= PRED_LEN,

        # Cechy znane w przyszłości: kalendarz + NWP (kluczowe!)
        time_varying_known_reals        = ["time_idx"] + KNOWN_FUTURE_REALS,
        time_varying_known_categoricals = KNOWN_FUTURE_CATS,

        # Cechy obserwowane tylko w historii
        time_varying_unknown_reals = TARGETS + OBSERVED_PAST,

        # Multi-target wymaga MultiNormalizer — osobny normalizer dla każdej zmiennej
        target_normalizer = MultiNormalizer([
            GroupNormalizer(groups=["group_id"], transformation="softplus")
            if target in ("precip_sum", "wind_max", "wind_mean")  # tylko nieujemne
            else GroupNormalizer(groups=["group_id"])
            for target in TARGETS
        ]),

        categorical_encoders = {
            "month":   NaNLabelEncoder(add_nan=True),
            "weekday": NaNLabelEncoder(add_nan=True),
        },

        allow_missing_timesteps = True,
    )
    return dataset


def train(years_back: int = DEFAULT_YEARS,
          epochs:     int = DEFAULT_EPOCHS,
          batch_size: int = DEFAULT_BATCH):

    print(f"[{datetime.now():%Y-%m-%d %H:%M}] === TRENING TFT 10-DNI ===")
    print(f"GPU: {'TAK — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NIE (CPU)'}") 
    torch.set_float32_matmul_precision("medium")  # Tensor Cores RTX

    # ── Dane ────────────────────────────────────
    print(f"\nPobieram dane ({years_back} lat)...")
    raw = fetch_training_data(years_back=years_back)
    df  = add_features(raw)

    # Zapisz dane do cache
    df.to_parquet(DATA_PATH)
    print(f"Dane: {len(df)} dni | {df['date'].min().date()} → {df['date'].max().date()}")

    if len(df) < 365:
        raise ValueError(f"Za mało danych: {len(df)} dni. Potrzeba min. 365.")

    # ── Datasety ────────────────────────────────
    print("\nBuduję datasety...")
    train_ds = build_dataset(df, is_train=True)

    # Walidacja — niezależny dataset z ostatnich 90 dni
    # (from_dataset z predict=True daje 1 próbkę — zamiast tego buduj osobno)
    val_cutoff = df["time_idx"].max() - 90
    val_df     = df[df["time_idx"] > val_cutoff - ENCODER_LEN].copy()
    val_ds     = TimeSeriesDataSet.from_dataset(
        train_ds,
        val_df,
        stop_randomization=True,
    )

    train_loader = train_ds.to_dataloader(train=True,  batch_size=batch_size, num_workers=0)
    val_loader   = val_ds.to_dataloader(  train=False, batch_size=batch_size, num_workers=0)
    print(f"Train: {len(train_ds)} próbek | Val: {len(val_ds)} próbek")

    # ── Model ────────────────────────────────────
    print("\nBuduję model TFT...")
    n_targets = len(TARGETS)

    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate           = LR,
        hidden_size             = HIDDEN_SIZE,
        attention_head_size     = ATTENTION_HEADS,
        dropout                 = DROPOUT,
        hidden_continuous_size  = HIDDEN_CONT_SIZE,
        loss                    = MultiLoss([QuantileLoss(quantiles=[0.1, 0.5, 0.9])] * len(TARGETS)),
        log_interval            = 10,
        reduce_on_plateau_patience = 8,
        optimizer               = "adamw",
    )

    n_params = sum(p.numel() for p in tft.parameters() if p.requires_grad)
    print(f"Parametry modelu: {n_params:,}")

    # ── Trening ──────────────────────────────────
    # TrainingContext — atomowa podmiana po sukcesie
    _tft_ctx      = TrainingContext(MODEL_DIR)
    _tft_ctx.__enter__()
    _tft_tmp_dir  = _tft_ctx.tmp_dir

    callbacks = [
        EarlyStopping(
            monitor   = "val_loss",
            patience  = 20,
            min_delta = 1e-4,
            mode      = "min",
        ),
        ModelCheckpoint(
            dirpath   = _tft_tmp_dir,
            filename  = "tft_weather",
            monitor   = "val_loss",
            save_top_k= 1,
            mode      = "min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs              = epochs,
        accelerator             = "gpu" if torch.cuda.is_available() else "cpu",
        devices                 = 1,
        gradient_clip_val       = 0.1,
        enable_progress_bar     = True,
        enable_model_summary    = False,
        log_every_n_steps       = 5,
        callbacks               = callbacks,
        default_root_dir        = _tft_tmp_dir,
    )

    print(f"\nTrening (max {epochs} epok, early stop patience=15)...")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\nNajlepszy checkpoint: {best_ckpt}")

    # ── Ewaluacja ────────────────────────────────
    print("\nEwaluacja na zbiorze walidacyjnym...")
    # PyTorch 2.6: weights_only=True domyślnie — allowlist klas pytorch_forecasting
    from pytorch_forecasting.data import encoders as _pf_enc
    _safe = [getattr(_pf_enc, n) for n in dir(_pf_enc)
             if not n.startswith("_") and isinstance(getattr(_pf_enc, n), type)]
    torch.serialization.add_safe_globals(_safe)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt, map_location="cpu")
    preds    = best_tft.predict(val_loader, return_y=True, trainer_kwargs={"accelerator": "cpu"})

    # Metryki per target
    print("\nRMSE per zmienna (mediana q50):")
    try:
        output = preds.output
        y_all  = preds.y[0]
        for i, target in enumerate(TARGETS):
            try:
                if isinstance(output, (list, tuple)):
                    pred_med = output[i][..., 1].cpu().numpy().flatten()
                else:
                    pred_med = output[..., i, 1].cpu().numpy().flatten()
                if isinstance(y_all, (list, tuple)):
                    y = y_all[i].cpu().numpy().flatten()
                elif y_all.ndim == 3:
                    y = y_all[:, :, i].cpu().numpy().flatten()
                else:
                    y = y_all.cpu().numpy().flatten()
                rmse = float(np.sqrt(np.nanmean((y - pred_med) ** 2)))
                print(f"  {target:<20} RMSE = {rmse:.2f}")
            except Exception as e:
                print(f"  {target:<20} blad: {e}")
    except Exception as e:
        print(f"  Blad ewaluacji: {e} — model zapisany OK")

    # ── Metadane ────────────────────────────────
    meta = {
        "trained_at":   datetime.now().isoformat(),
        "best_ckpt":    best_ckpt,
        "targets":      TARGETS,
        "encoder_len":  ENCODER_LEN,
        "pred_len":     PRED_LEN,
        "epochs_done":  trainer.current_epoch,
        "val_loss":     float(trainer.checkpoint_callback.best_model_score or 0),
        "n_params":     n_params,
        "years_back":   years_back,
        "n_days":       len(df),
    }
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    # Atomowa podmiana — PO ewaluacji zeby lapac najnowszy checkpoint
    _tft_ctx.__exit__(None, None, None)
    print("Model TFT podmieniony atomowo")
    print(f"\nMetadane: {meta_path}")
    print(f"Model:    {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",  type=int, default=DEFAULT_YEARS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",  type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()
    train(years_back=args.years, epochs=args.epochs, batch_size=args.batch)