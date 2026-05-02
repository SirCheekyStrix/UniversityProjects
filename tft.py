import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import matplotlib.pyplot as plt

from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss, RMSE
from pytorch_forecasting.data import GroupNormalizer

# --- KONFIGURACJA ---
# Skupiamy się TYLKO na temperaturze jako celu
TARGET = "atm_temperature_C"

# Wszystkie zmienne, które model będzie analizował (historia)
ALL_WEATHER_VARS = [
    "atm_temperature_C", "humidity_percent", "pressure_hPa", 
    "wind_speed", "wind_gust", "rain_rate"
]

ENCODER_DAYS = 14
PRED_DAYS = 10
HOURS_IN_DAY = 24

def prepare_open_meteo(df: pd.DataFrame):
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    
    # Agregacja
    df = df.groupby("timestamp").agg({
        "temperature": "mean", "humidity": "mean", "pressure": "mean",
        "wind_speed": "mean", "wind_direction": "mean", "wind_gust": "mean", "rain_rate": "sum"
    }).reset_index()

    df = df.rename(columns={
        "temperature": "atm_temperature_C", "humidity": "humidity_percent",
        "pressure": "pressure_hPa", "wind_direction": "wind_direction"
    })
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))
    
    # Dane kalendarzowe
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month"] = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday"] = df["timestamp"].dt.weekday
    df["station_id"] = 0
    
    # Proste skalowanie, żeby wartości były blisko zera i jedynki
    df["atm_temperature_C"] = df["atm_temperature_C"] / 20.0  # ok. -1 do +2 (dla -20 do +40 st)
    df["pressure_hPa"] = (df["pressure_hPa"] - 1013) / 10.0   # odchyłka od średniego ciśnienia
    df["humidity_percent"] = df["humidity_percent"] / 100.0   # zakres 0-1

    # Interpolacja braków dla wszystkich zmiennych pogodowych
    df[ALL_WEATHER_VARS] = df[ALL_WEATHER_VARS].interpolate().bfill().ffill().astype(np.float32)
    return df

def build_datasets(df: pd.DataFrame):
    max_encoder_length = ENCODER_DAYS * HOURS_IN_DAY
    max_prediction_length = PRED_DAYS * HOURS_IN_DAY
    training_cutoff = df["time_idx"].max() - max_prediction_length

    # Transformation=None mówi: "Nie dotykaj moich liczb"
    target_normalizer = GroupNormalizer(groups=["station_id"], transformation=None)

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=TARGET,
        group_ids=["station_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        # Dodajemy nowe cechy cykliczne tutaj:
        time_varying_known_reals=["hour_sin", "hour_cos", "month_sin", "month_cos", "weekday"],
        time_varying_unknown_reals=ALL_WEATHER_VARS,
        target_normalizer=target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True)    
    return training, validation

def train_model(training, validation):
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=15)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=15)

    # Uproszczony model dla pojedynczego celu
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=16,          # Zmniejszone, by uniknąć overfittingu dla jednego celu
        attention_head_size=2,
        dropout=0.1,             # Większy dropout dla lepszej generalizacji
        hidden_continuous_size=16,
        output_size=1,           # Zwykła siódemka (7 kwantyli), bez listy
        loss=RMSE(),     # Zwykły loss, bez MultiLoss
        log_interval=10,
        reduce_on_plateau_patience=4
    )

    torch.set_float32_matmul_precision('medium')

    early_stop = EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop, lr_monitor],
        gradient_clip_val=0.1
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    return tft

def predict_10_days(model, df):
    predictions = model.predict(df, mode="prediction")
    
    # Mamy tylko jeden target, więc predictions to od razu tensor z wynikami
    res = predictions[0].cpu().numpy()
        
    last_date = df["timestamp"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=PRED_DAYS*HOURS_IN_DAY, freq="h")
    
    forecast_df = pd.DataFrame({
        "timestamp": future_dates,
        TARGET: res
    })
    
    return forecast_df

def show_forecast(forecast_df):
    plt.figure(figsize=(12, 6))
    forecast_df[TARGET] = forecast_df[TARGET] * 20.0  # Odwracamy skalowanie temperatury
    plt.plot(forecast_df["timestamp"], forecast_df[TARGET], 
             label="Prognoza Temperatury (Mediana)", color='tomato', linewidth=2)
    
    plt.title("Prognoza: Temperatura atm_temperature_C")
    plt.ylabel("Wartość [°C]")
    plt.xlabel("Data i godzina")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=35)

    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_parquet("data/station_timeseries_tft.parquet")
    df = prepare_open_meteo(df)

    training, validation = build_datasets(df)
    model = train_model(training, validation)

    # Predykcja
    forecast = predict_10_days(model, df)
    
    # Zapis do CSV
    forecast.to_csv("prognoza_temperatury_10dni.csv", index=False, sep=';', decimal=',')
    print("Prognoza zapisana do pliku prognoza_temperatury_10dni.csv")

    # Wyświetlenie wykresu
    show_forecast(forecast)

if __name__ == "__main__":
    main()