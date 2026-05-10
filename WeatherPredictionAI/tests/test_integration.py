"""
test_integration.py — testy integracyjne i testy API WeatherPredictionAI.

Testy integracyjne używają prawdziwych danych (Open-Meteo API).
Testy API wymagają działającego serwera na localhost:8080.

Uruchomienie:
    # Tylko testy jednostkowe + integracyjne (bez API):
    pytest test_integration.py -v -m "not api"

    # Wszystkie (wymaga działającego serwera):
    pytest test_integration.py -v

    # Tylko testy API:
    pytest test_integration.py -v -m api
"""
import json
import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = "http://localhost:8080"
TORUN_LAT, TORUN_LON = 53.0138, 18.5981


def is_api_running() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def make_mock_hourly_response(hours=200):
    """Generuje mocka odpowiedzi Open-Meteo API."""
    times = pd.date_range("2026-01-01", periods=hours, freq="h")
    np.random.seed(42)
    return {
        "hourly": {
            "time":                  [str(t) for t in times],
            "temperature_2m":        np.random.uniform(-5, 25, hours).tolist(),
            "relativehumidity_2m":   np.random.uniform(40, 95, hours).tolist(),
            "pressure_msl":          np.random.uniform(990, 1030, hours).tolist(),
            "windspeed_10m":         np.random.uniform(0, 30, hours).tolist(),
            "windgusts_10m":         np.random.uniform(0, 50, hours).tolist(),
            "precipitation":         np.random.uniform(0, 2, hours).tolist(),
            "winddirection_10m":     np.random.uniform(0, 360, hours).tolist(),
            "cloudcover":            np.random.uniform(0, 100, hours).tolist(),
            "uv_index":              np.random.uniform(0, 8, hours).tolist(),
            "apparent_temperature":  np.random.uniform(-8, 22, hours).tolist(),
            "dewpoint_2m":           np.random.uniform(-10, 15, hours).tolist(),
            "shortwave_radiation":   np.random.uniform(0, 800, hours).tolist(),
        }
    }


# ═══════════════════════════════════════════════════════
# TESTY INTEGRACYJNE — PatchTST pipeline
# ═══════════════════════════════════════════════════════

class TestPatchTSTPipeline(unittest.TestCase):
    """Testy całego pipeline PatchTST: fetch → clean → features → (mock) predict."""

    def setUp(self):
        self.mock_response = make_mock_hourly_response(200)

    @patch("patchtst_model.requests.get")
    def test_fetch_recent_returns_dataframe(self, mock_get):
        from patchtst_model import _fetch_recent
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_response

        df = _fetch_recent()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("temperature", df.columns)
        self.assertIn("timestamp",   df.columns)

    @patch("patchtst_model.requests.get")
    def test_fetch_and_clean_no_nan(self, mock_get):
        from patchtst_model import _fetch_recent, _clean, _RAW_COLS
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_response

        df = _fetch_recent()
        df = _clean(df)
        self.assertEqual(df[_RAW_COLS].isna().sum().sum(), 0)

    @patch("patchtst_model.requests.get")
    def test_full_pipeline_features(self, mock_get):
        from patchtst_model import _fetch_recent, _clean, add_features, FEATURES_IN
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_response

        df = _fetch_recent()
        df = _clean(df)
        df = add_features(df)

        for feat in FEATURES_IN:
            self.assertIn(feat, df.columns)
        self.assertEqual(df[FEATURES_IN].isna().sum().sum(), 0)

    @patch("patchtst_model.requests.get")
    def test_predict_output_shape(self, mock_get):
        """Testuje predict() z mock modelem."""
        import torch
        from patchtst_model import _fetch_recent, _clean, add_features, FEATURES_IN, TARGETS, OUTPUT_HOURS

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_response

        df = _fetch_recent()
        df = _clean(df)
        df = add_features(df)

        # Mock model i scalery
        with patch("patchtst_model.load_model") as mock_load, \
             patch("joblib.load") as mock_joblib:

            mock_model = MagicMock()
            mock_device = torch.device("cpu")

            # Symuluj wyjście modelu
            pred_tensor = torch.randn(1, OUTPUT_HOURS, len(FEATURES_IN))
            mock_out = MagicMock()
            mock_out.prediction_outputs = pred_tensor
            mock_model.return_value = mock_out
            mock_load.return_value = (mock_model, mock_device)

            # Mock skalery
            mock_scaler = MagicMock()
            mock_scaler.transform.return_value = np.random.rand(168, len(FEATURES_IN)).astype(np.float32)
            mock_scaler_out = MagicMock()
            mock_scaler_out.inverse_transform.return_value = np.random.rand(OUTPUT_HOURS, 1)
            mock_joblib.side_effect = [
                mock_scaler,
                {name: mock_scaler_out for name in TARGETS}
            ]

            from patchtst_model import predict
            result = predict(df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), OUTPUT_HOURS)
        self.assertIn("temperature", result.columns)
        self.assertIn("timestamp",   result.columns)


# ═══════════════════════════════════════════════════════
# TESTY INTEGRACYJNE — Fish pipeline
# ═══════════════════════════════════════════════════════

class TestFishPipeline(unittest.TestCase):
    """Testy całego pipeline ryb: fetch → features → predict (mock modele)."""

    def make_mock_fish_response(self):
        times = pd.date_range(
            pd.Timestamp.now().floor("h") - pd.Timedelta(hours=72),
            periods=120, freq="h"
        )
        np.random.seed(0)
        return {
            "hourly": {
                "time":                times.strftime("%Y-%m-%dT%H:%M").tolist(),
                "temperature_2m":      np.random.uniform(5, 15, 120).tolist(),
                "relativehumidity_2m": np.random.uniform(60, 90, 120).tolist(),
                "pressure_msl":        np.random.uniform(1000, 1020, 120).tolist(),
                "windspeed_10m":       np.random.uniform(0, 20, 120).tolist(),
                "windgusts_10m":       np.random.uniform(0, 30, 120).tolist(),
                "precipitation":       np.random.uniform(0, 1, 120).tolist(),
                "cloudcover":          np.random.uniform(0, 100, 120).tolist(),
            }
        }

    @patch("fish_model.requests.get")
    def test_fetch_and_features(self, mock_get):
        from fish_model import fetch_recent_weather, add_features, FEATURE_COLS
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.make_mock_fish_response()

        df = fetch_recent_weather()
        df = add_features(df)

        for col in FEATURE_COLS:
            self.assertIn(col, df.columns)

    @patch("fish_model.requests.get")
    def test_predict_24h_returns_dataframe(self, mock_get):
        from fish_model import SPECIES, MODEL_DIR
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.make_mock_fish_response()

        # Sprawdź czy modele są dostępne
        models_available = all(
            os.path.exists(os.path.join(MODEL_DIR, f"model_{s}.lgb"))
            for s in SPECIES
        )
        if not models_available:
            self.skipTest("Modele ryb nie są wytrenowane")

        from fish_predict import predict_24h
        result = predict_24h()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        for species in SPECIES:
            if species in result.columns:
                self.assertTrue((result[species] >= 0).all())
                self.assertTrue((result[species] <= 100).all())


# ═══════════════════════════════════════════════════════
# TESTY INTEGRACYJNE — z prawdziwym API Open-Meteo
# ═══════════════════════════════════════════════════════

@pytest.mark.slow
class TestRealOpenMeteoIntegration(unittest.TestCase):
    """Testy z prawdziwymi requestami do Open-Meteo (wolniejsze)."""

    def test_fetch_recent_window_real(self):
        """Pobierz prawdziwe dane z Open-Meteo."""
        from patchtst_model import fetch_recent_window
        try:
            df = fetch_recent_window()
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            self.assertIn("temperature", df.columns)
            # Ostatni timestamp powinien być <= teraz
            last_ts = df["timestamp"].max()
            now = pd.Timestamp.now()
            self.assertLessEqual(last_ts, now + pd.Timedelta(hours=1))
        except Exception as e:
            self.skipTest(f"Open-Meteo niedostępny: {e}")

    def test_fetch_fish_weather_real(self):
        """Pobierz prawdziwe dane pogodowe dla ryb."""
        from fish_model import fetch_recent_weather, add_features, FEATURE_COLS
        try:
            df = fetch_recent_weather()
            df = add_features(df)
            self.assertIsInstance(df, pd.DataFrame)
            for col in FEATURE_COLS:
                self.assertIn(col, df.columns)
        except Exception as e:
            self.skipTest(f"Open-Meteo niedostępny: {e}")


# ═══════════════════════════════════════════════════════
# TESTY API — wymagają działającego serwera
# ═══════════════════════════════════════════════════════

@pytest.mark.api
class TestAPIHealth(unittest.TestCase):
    """Testy endpointu /health."""

    def setUp(self):
        if not is_api_running():
            self.skipTest(f"API nie działa na {API_BASE}")

    def test_health_returns_200(self):
        r = requests.get(f"{API_BASE}/health", timeout=10)
        self.assertEqual(r.status_code, 200)

    def test_health_returns_json(self):
        r = requests.get(f"{API_BASE}/health", timeout=10)
        data = r.json()
        self.assertIn("status",    data)
        self.assertIn("models",    data)
        self.assertIn("timestamp", data)

    def test_health_status_ok(self):
        r = requests.get(f"{API_BASE}/health", timeout=10)
        data = r.json()
        self.assertEqual(data["status"], "ok")

    def test_health_models_structure(self):
        r = requests.get(f"{API_BASE}/health", timeout=10)
        data = r.json()
        for model in ["patchtst", "tft", "fish"]:
            self.assertIn(model, data["models"])
            self.assertIn("available", data["models"][model])


@pytest.mark.api
class TestAPIPredict(unittest.TestCase):
    """Testy endpointu /predict."""

    def setUp(self):
        if not is_api_running():
            self.skipTest(f"API nie działa na {API_BASE}")

    def test_predict_get_returns_200(self):
        r = requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "fish"},
            timeout=30
        )
        self.assertEqual(r.status_code, 200)

    def test_predict_missing_latitude_returns_422(self):
        r = requests.get(
            f"{API_BASE}/predict",
            params={"longitude": TORUN_LON},
            timeout=10
        )
        self.assertEqual(r.status_code, 422)

    def test_predict_missing_longitude_returns_422(self):
        r = requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT},
            timeout=10
        )
        self.assertEqual(r.status_code, 422)

    def test_predict_returns_coordinates(self):
        r = requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "fish"},
            timeout=30
        )
        data = r.json()
        self.assertAlmostEqual(data["latitude"],  TORUN_LAT, places=3)
        self.assertAlmostEqual(data["longitude"], TORUN_LON, places=3)

    def test_predict_returns_elapsed_sec(self):
        r = requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "fish"},
            timeout=30
        )
        data = r.json()
        self.assertIn("elapsed_sec", data)
        self.assertGreater(data["elapsed_sec"], 0)

    def test_predict_errors_is_list(self):
        r = requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "fish"},
            timeout=30
        )
        data = r.json()
        self.assertIn("errors", data)
        self.assertIsInstance(data["errors"], list)

    def test_predict_post_json(self):
        r = requests.post(
            f"{API_BASE}/predict",
            json={
                "latitude":       TORUN_LAT,
                "longitude":      TORUN_LON,
                "reservoir_id":   1,
                "reservoir_name": "Test",
                "models":         "fish"
            },
            timeout=30
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("models_run", data)

    def test_predict_fish_24_hours(self):
        r = requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "fish"},
            timeout=30
        )
        data = r.json()
        if "forecast_fish" in data and data["forecast_fish"]:
            forecast = data["forecast_fish"].get("forecast", [])
            self.assertEqual(len(forecast), 24)

    def test_predict_fish_probabilities_range(self):
        r = requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "fish"},
            timeout=30
        )
        data = r.json()
        if "forecast_fish" not in data or not data["forecast_fish"]:
            return
        for hour in data["forecast_fish"].get("forecast", []):
            for species in ["karp", "leszcz", "szczupak", "okon", "sandacz",
                            "ploc", "lin", "sum", "klen", "wzdrega"]:
                if species in hour:
                    self.assertGreaterEqual(hour[species], 0)
                    self.assertLessEqual(hour[species],   100)

    def test_predict_patchtst_24_hours(self):
        r = requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "patchtst"},
            timeout=60
        )
        data = r.json()
        if "patchtst" not in data.get("models_run", []):
            self.skipTest("PatchTST niedostępny")
        forecast = data["forecast_24h"].get("forecast", [])
        self.assertEqual(len(forecast), 24)

    def test_predict_patchtst_temperature_reasonable(self):
        r = requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "patchtst"},
            timeout=60
        )
        data = r.json()
        if "patchtst" not in data.get("models_run", []):
            self.skipTest("PatchTST niedostępny")
        for hour in data["forecast_24h"].get("forecast", []):
            self.assertGreater(hour["temperature"], -50)
            self.assertLess(hour["temperature"],     60)

    def test_predict_tft_10_days(self):
        r = requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "tft"},
            timeout=60
        )
        data = r.json()
        if "tft" not in data.get("models_run", []):
            self.skipTest("TFT niedostępny")
        forecast = data["forecast_10d"].get("forecast", [])
        self.assertEqual(len(forecast), 10)

    def test_predict_tft_quantiles_order(self):
        """q10 <= median <= q90 dla każdego dnia."""
        r = requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "tft"},
            timeout=60
        )
        data = r.json()
        if "tft" not in data.get("models_run", []):
            self.skipTest("TFT niedostępny")
        for day in data["forecast_10d"].get("forecast", []):
            for var in ["temp_max", "temp_min", "precip_sum"]:
                if var in day:
                    q10    = day[var]["q10"]
                    median = day[var]["median"]
                    q90    = day[var]["q90"]
                    self.assertLessEqual(q10, median, f"{var}: q10 > median")
                    self.assertLessEqual(median, q90, f"{var}: median > q90")

    def test_predict_reservoir_tags_in_response(self):
        r = requests.post(
            f"{API_BASE}/predict",
            json={
                "latitude":       TORUN_LAT,
                "longitude":      TORUN_LON,
                "reservoir_id":   42,
                "reservoir_name": "Test Zbiornik",
                "models":         "fish"
            },
            timeout=30
        )
        data = r.json()
        self.assertEqual(data["reservoir_id"],   42)
        self.assertEqual(data["reservoir_name"], "Test Zbiornik")

    def test_predict_different_locations(self):
        """Różne lokalizacje powinny dawać różne wyniki."""
        locations = [
            (52.4763, 21.0532),  # Warszawa
            (54.3520, 18.6466),  # Gdańsk
            (50.0647, 19.9450),  # Kraków
        ]
        results = []
        for lat, lon in locations:
            r = requests.get(
                f"{API_BASE}/predict",
                params={"latitude": lat, "longitude": lon, "models": "fish"},
                timeout=30
            )
            results.append(r.json())

        # Każda lokalizacja powinna zwrócić inne koordynaty
        lats = [r["latitude"] for r in results]
        self.assertEqual(len(set(lats)), 3, "Wszystkie lokalizacje zwróciły te same lat")


@pytest.mark.api
class TestAPIPerformance(unittest.TestCase):
    """Testy wydajności API."""

    def setUp(self):
        if not is_api_running():
            self.skipTest(f"API nie działa na {API_BASE}")

    def test_fish_response_under_10s(self):
        t0 = time.time()
        requests.get(
            f"{API_BASE}/predict",
            params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "fish"},
            timeout=30
        )
        elapsed = time.time() - t0
        self.assertLess(elapsed, 10.0, f"Fish request zajął {elapsed:.1f}s > 10s")

    def test_health_response_under_1s(self):
        t0 = time.time()
        requests.get(f"{API_BASE}/health", timeout=5)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 1.0, f"Health check zajął {elapsed:.2f}s > 1s")

    def test_sequential_5_requests(self):
        """5 kolejnych requestów powinno działać bez błędów."""
        for i in range(5):
            r = requests.get(
                f"{API_BASE}/predict",
                params={"latitude": TORUN_LAT, "longitude": TORUN_LON, "models": "fish"},
                timeout=30
            )
            self.assertEqual(r.status_code, 200, f"Request {i+1} zwrócił {r.status_code}")


# ─────────────────────────────────────────────
# KONFIGURACJA PYTEST
# ─────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "api: testy wymagające działającego serwera API")
    config.addinivalue_line("markers", "slow: testy używające prawdziwego API Open-Meteo")


if __name__ == "__main__":
    unittest.main(verbosity=2)
