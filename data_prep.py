import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


FEATURE_COLS = [
    "humidity_percent",
    "pressure_hPa",
    "wind_speed",
    "wind_direction",
    "wind_gust",
    "rain_rate",
    "solar_radiation",
    "water_temperature_C",
    "water_level_cm",
    "water_pH",
    "water_conductivity",
    "water_oxygen",
    "moon_phase",
    "moon_illumination",
    "hour",
    "day",
    "month",
    "weekday"
]


def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["bite_probability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(50)
        ]
    )

    return model, X_test, y_test
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mask = np.isfinite(y_test) & np.isfinite(y_pred)

    rmse = np.sqrt(mean_squared_error(
        y_test[mask],
        y_pred[mask]
    ))

    print(f"RMSE: {rmse:.2f}")

    return pd.DataFrame({
        "y_true": y_test[mask],
        "y_pred": y_pred[mask]
    })

def feature_importance(model):
    return pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)
def predict_from_row(model, row: dict) -> float:
    """
    row: dict zawierający dokładnie FEATURE_COLS
    """
    X = pd.DataFrame([row])[FEATURE_COLS]
    return float(model.predict(X)[0])
def predict_from_timestamp(model, ts: str, record: dict):
    """
    ts: ISO timestamp
    record: dict z pogodą + wodą + księżycem (bez bite)
    """
    ts = pd.to_datetime(ts)

    record["hour"] = ts.hour
    record["day"] = ts.day
    record["month"] = ts.month
    record["weekday"] = ts.weekday()

    return predict_from_row(model, record)
df = pd.read_parquet("data/station_timeseries.parquet")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["weekday"] = df["timestamp"].dt.weekday
print(">>> first 5 timestamps:", df["timestamp"].head())

# USUŃ rekordy z NaN w feature'ach lub target
df = df.dropna(subset=FEATURE_COLS + ["bite_probability"])

model, X_test, y_test = train_model(df)

pred_df = evaluate_model(model, X_test, y_test)
print(pred_df.head())

imp = feature_importance(model)
pred = predict_from_row(model, {
    "humidity_percent": 80,
    "pressure_hPa": 1010,
    "wind_speed": 5,
    "wind_direction": 180,
    "wind_gust": 10,
    "rain_rate": 0,
    "solar_radiation": 500,
    "water_temperature_C": 20,
    "water_level_cm": 50,
    "water_pH": 7,
    "water_conductivity": 300,
    "water_oxygen": 8,
    "moon_phase": 0.5,
    "moon_illumination": 50,
    "hour": 14,
    "day": 15,
    "month": 6,
    "weekday": 5
})
print(f"Predicted bite probability: {pred:.1f}%")
print(imp.head(10))
