"""
api_server.py — REST API do predykcji pogody i ryb.

Endpoint:
    POST /predict
    Body: {"latitude": 53.01, "longitude": 18.59, "reservoir_id": 1, "reservoir_name": "Zbiornik"}
    Response: prognoza 24h + 10d + ryby + zapis do InfluxDB

    GET /predict?latitude=53.01&longitude=18.59
    Response: jw.

    GET /health
    Response: {"status": "ok", "models": {...}}

Uruchomienie:
    pip install fastapi uvicorn
    python api_server.py
    # lub:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("api_server")

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Zainstaluj: pip install fastapi uvicorn")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

app = FastAPI(
    title="WeatherPredictionAI API",
    description="Prognoza pogody 24h (PatchTST) + 10d (TFT) + branie ryb (LightGBM)",
    version="2.0",
)

# ─────────────────────────────────────────────
# MODELE REQUEST / RESPONSE
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    latitude:       float
    longitude:      float
    reservoir_id:   Optional[int]  = None
    reservoir_name: Optional[str]  = None
    owner:          Optional[str]  = None
    models:         Optional[str]  = "patchtst,tft,fish"


# ─────────────────────────────────────────────
# HELPERY
# ─────────────────────────────────────────────

def _check_model(path: str) -> dict:
    if os.path.exists(path):
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return {"available": True, "trained_at": mtime.isoformat()[:19]}
    return {"available": False, "trained_at": None}


def _write_to_influx_with_tags(result_dict, measurement, reservoir_id, reservoir_name, owner, lat, lon):
    try:
        from influxdb_client import InfluxDBClient, Point, WritePrecision
        from influxdb_client.client.write_api import SYNCHRONOUS
        from datetime import timezone

        INFLUX_URL    = "http://localhost:8086"
        INFLUX_TOKEN  = "mH1sJUpajjdlKcQrM64YVu8efBZymCm--X0Jp2nRoaHFIZduitvapbZATXrA6t2TxnwQ2EVJ8RuxrfNM4efeDA=="
        INFLUX_ORG    = "zlote-branie"
        INFLUX_BUCKET = "predictions"

        client    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        points    = []
        gen_at    = result_dict.get("generated_at", datetime.now().isoformat())[:19]

        for row in result_dict.get("forecast", []):
            ts_str = row.get("timestamp") or row.get("date")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            p = Point(measurement).time(ts, WritePrecision.S)
            if reservoir_id   is not None: p = p.tag("reservoir_id",   str(reservoir_id))
            if reservoir_name is not None: p = p.tag("reservoir_name", str(reservoir_name))
            if owner          is not None: p = p.tag("owner",          str(owner))
            p = p.tag("lat",          f"{lat:.4f}")
            p = p.tag("lon",          f"{lon:.4f}")
            p = p.tag("generated_at", gen_at)
            p = p.tag("model",        result_dict.get("model", "unknown"))

            skip = {"timestamp", "date", "generated_at", "day"}
            for field, val in row.items():
                if field in skip:
                    continue
                if isinstance(val, dict):
                    for suffix, v in val.items():
                        if v is not None:
                            try: p = p.field(f"{field}_{suffix}", float(v))
                            except: pass
                else:
                    try: p = p.field(field, float(val))
                    except: pass

            points.append(p)

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)
        client.close()
        log.info(f"InfluxDB [{measurement}]: {len(points)} punktow")
        return len(points)
    except Exception as e:
        log.warning(f"InfluxDB blad [{measurement}]: {e}")
        return 0


def _run_patchtst(lat, lon):
    import patchtst_model as pm
    import pandas as pd
    orig_lat, orig_lon = pm.LAT, pm.LON
    pm.LAT, pm.LON = lat, lon
    try:
        raw = pm._fetch_recent()
        raw = raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        raw = pm._clean(raw)
        now_utc = pd.Timestamp.utcnow().floor("h").tz_localize(None)
        raw = raw[raw["timestamp"] <= now_utc].reset_index(drop=True)
        df  = pm.add_features(raw)
        result_df = pm.predict(df)

        forecast_list = []
        for _, row in result_df.iterrows():
            entry = {"timestamp": str(row["timestamp"])}
            for col in result_df.columns:
                if col != "timestamp":
                    entry[col] = round(float(row[col]), 2)
            forecast_list.append(entry)

        return {
            "generated_at": datetime.now().isoformat(),
            "model":        "PatchTST",
            "latitude":     lat,
            "longitude":    lon,
            "forecast":     forecast_list,
        }
    finally:
        pm.LAT, pm.LON = orig_lat, orig_lon


def _run_tft(lat, lon):
    import tft_model as tm
    orig_lat, orig_lon = tm.LAT, tm.LON
    tm.LAT, tm.LON = lat, lon
    try:
        from tft_predict import predict_10days
        return predict_10days()
    finally:
        tm.LAT, tm.LON = orig_lat, orig_lon


def _run_fish(lat, lon, weather_df=None):
    try:
        from fish_predict import predict_24h
        return predict_24h(df_input=weather_df)
    except Exception as e:
        log.warning(f"Fish predict blad: {e}")
        return None


# ─────────────────────────────────────────────
# ENDPOINTY
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "timestamp": datetime.now().isoformat()[:19],
        "models": {
            "patchtst": _check_model(os.path.join(BASE_DIR, "patchtst_files", "patchtst_weather.pth")),
            "tft":      _check_model(os.path.join(BASE_DIR, "tft_model_files", "tft_weather_best.ckpt")),
            "fish":     _check_model(os.path.join(BASE_DIR, "fish_model_files", "fish_model.pkl")),
        }
    }


@app.get("/predict")
def predict_get(
    latitude:       float          = Query(...),
    longitude:      float          = Query(...),
    reservoir_id:   Optional[int]  = Query(None),
    reservoir_name: Optional[str]  = Query(None),
    owner:          Optional[str]  = Query(None),
    models:         Optional[str]  = Query("patchtst,tft,fish"),
):
    return _predict(latitude, longitude, reservoir_id, reservoir_name, owner, models)


@app.post("/predict")
def predict_post(req: PredictRequest):
    return _predict(req.latitude, req.longitude,
                    req.reservoir_id, req.reservoir_name, req.owner,
                    req.models or "patchtst,tft,fish")


def _predict(lat, lon, reservoir_id, reservoir_name, owner, models_str):
    t0         = time.time()
    models_req = [m.strip().lower() for m in (models_str or "patchtst,tft,fish").split(",")]
    response   = {
        "generated_at":   datetime.now().isoformat()[:19],
        "latitude":       lat,
        "longitude":      lon,
        "reservoir_id":   reservoir_id,
        "reservoir_name": reservoir_name,
        "models_run":     [],
        "errors":         [],
    }

    weather_df = None

    # ── PatchTST 24h ─────────────────────────────────────────────────────
    if "patchtst" in models_req:
        try:
            log.info(f"PatchTST predict: lat={lat}, lon={lon}")
            result = _run_patchtst(lat, lon)
            response["forecast_24h"] = result
            response["models_run"].append("patchtst")
            _write_to_influx_with_tags(result, "forecast_weather_24h",
                                       reservoir_id, reservoir_name, owner, lat, lon)
            import pandas as pd
            rows = result["forecast"]
            weather_df = pd.DataFrame(rows)
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
            # Przemapuj nazwy kolumn PatchTST -> fish_model
            weather_df = weather_df.rename(columns={
                "temperature":  "temperature_c",
                "humidity":     "humidity_pct",
                "pressure":     "pressure_hpa",
                "wind_speed":   "wind_speed_kmh",
                "wind_gust":    "wind_gust_kmh",
                "rain_rate":    "rain_rate_mm",
                "cloudcover":   "cloudcover_pct",
            })
        except Exception as e:
            log.error(f"PatchTST blad: {e}")
            response["errors"].append({"model": "patchtst", "error": str(e)})

    # ── TFT 10d ──────────────────────────────────────────────────────────
    if "tft" in models_req:
        try:
            log.info(f"TFT predict: lat={lat}, lon={lon}")
            result = _run_tft(lat, lon)
            response["forecast_10d"] = result
            response["models_run"].append("tft")
            _write_to_influx_with_tags(result, "forecast_weather_10d",
                                       reservoir_id, reservoir_name, owner, lat, lon)
        except Exception as e:
            log.error(f"TFT blad: {e}")
            response["errors"].append({"model": "tft", "error": str(e)})

    # ── Fish ─────────────────────────────────────────────────────────────
    if "fish" in models_req:
        try:
            log.info(f"Fish predict: lat={lat}, lon={lon}")
            result = _run_fish(lat, lon, weather_df)
            if result is not None and (not hasattr(result, "empty") or not result.empty):
                import pandas as pd
                if isinstance(result, pd.DataFrame):
                    fish_dict = {
                        "forecast": result.to_dict(orient="records"),
                        "generated_at": datetime.now().isoformat(),
                        "model": "LightGBM"
                    }
                else:
                    fish_dict = result
                response["forecast_fish"] = fish_dict
                response["models_run"].append("fish")
                _write_to_influx_with_tags(fish_dict, "forecast_fish_24h",
                                           reservoir_id, reservoir_name, owner, lat, lon)
        except Exception as e:
            import traceback; log.error(f"Fish blad: {e}\n{traceback.format_exc()}")
            response["errors"].append({"model": "fish", "error": str(e)})

    response["elapsed_sec"] = round(time.time() - t0, 2)
    return JSONResponse(content=response)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    log.info(f"Uruchamiam API server na {args.host}:{args.port}")
    uvicorn.run("api_server:app", host=args.host, port=args.port,
                reload=args.reload, workers=1, timeout_keep_alive=30)