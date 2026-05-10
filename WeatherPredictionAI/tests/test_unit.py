"""
test_unit.py — testy jednostkowe wszystkich modułów WeatherPredictionAI.

Uruchomienie:
    pytest test_unit.py -v
    pytest test_unit.py -v --tb=short
    python -m pytest test_unit.py -v
"""
import math
import os
import sys
import unittest
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Dodaj katalog projektu do PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────
# HELPERY — generowanie danych testowych
# ─────────────────────────────────────────────

def make_weather_df(hours=200, start="2026-01-01"):
    """Generuje minimalny DataFrame pogodowy dla testów."""
    ts = pd.date_range(start, periods=hours, freq="h")
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp":          ts,
        "temperature":        np.random.uniform(-5, 25, hours),
        "humidity":           np.random.uniform(40, 95, hours),
        "pressure":           np.random.uniform(990, 1030, hours),
        "wind_speed":         np.random.uniform(0, 30, hours),
        "wind_gust":          np.random.uniform(0, 50, hours),
        "rain_rate":          np.random.uniform(0, 2, hours),
        "wind_dir_sin":       np.random.uniform(-1, 1, hours),
        "wind_dir_cos":       np.random.uniform(-1, 1, hours),
        "cloudcover":         np.random.uniform(0, 100, hours),
        "apparent_temp":      np.random.uniform(-8, 22, hours),
        "dewpoint":           np.random.uniform(-10, 15, hours),
        "shortwave_radiation":np.random.uniform(0, 800, hours),
        "uv_index":           np.random.uniform(0, 8, hours),
    })


def make_fish_df(hours=50, start="2026-04-01"):
    """Generuje DataFrame z kolumnami wymaganymi przez fish_model."""
    ts = pd.date_range(start, periods=hours, freq="h")
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp":      ts,
        "temperature_c":  np.random.uniform(5, 20, hours),
        "humidity_pct":   np.random.uniform(50, 90, hours),
        "pressure_hpa":   np.random.uniform(995, 1025, hours),
        "wind_speed_kmh": np.random.uniform(0, 25, hours),
        "wind_gust_kmh":  np.random.uniform(0, 40, hours),
        "rain_rate_mm":   np.random.uniform(0, 1, hours),
        "cloudcover_pct": np.random.uniform(0, 100, hours),
    })


# ═══════════════════════════════════════════════════════
# TESTY: patchtst_model.py
# ═══════════════════════════════════════════════════════

class TestPatchTSTFeatures(unittest.TestCase):
    """Testy inżynierii cech w patchtst_model."""

    def setUp(self):
        from patchtst_model import add_features
        self.add_features = add_features
        self.df = make_weather_df(200)

    def test_add_features_returns_dataframe(self):
        result = self.add_features(self.df.copy())
        self.assertIsInstance(result, pd.DataFrame)

    def test_add_features_adds_hour_sin_cos(self):
        result = self.add_features(self.df.copy())
        self.assertIn("hour_sin", result.columns)
        self.assertIn("hour_cos", result.columns)

    def test_hour_sin_range(self):
        result = self.add_features(self.df.copy())
        self.assertTrue((result["hour_sin"] >= -1).all())
        self.assertTrue((result["hour_sin"] <= 1).all())

    def test_hour_cos_range(self):
        result = self.add_features(self.df.copy())
        self.assertTrue((result["hour_cos"] >= -1).all())
        self.assertTrue((result["hour_cos"] <= 1).all())

    def test_add_features_adds_doy(self):
        result = self.add_features(self.df.copy())
        self.assertIn("doy_sin", result.columns)
        self.assertIn("doy_cos", result.columns)

    def test_pressure_diff_columns(self):
        result = self.add_features(self.df.copy())
        self.assertIn("d_pressure_3h",  result.columns)
        self.assertIn("d_pressure_24h", result.columns)

    def test_temp_daily_max_gte_min(self):
        result = self.add_features(self.df.copy())
        self.assertTrue((result["temp_daily_max"] >= result["temp_daily_min"]).all())

    def test_solar_heating_non_negative(self):
        result = self.add_features(self.df.copy())
        self.assertTrue((result["solar_heating"] >= 0).all())

    def test_hour_shifted_range(self):
        result = self.add_features(self.df.copy())
        self.assertTrue((result["hour_shifted"] >= -1).all())
        self.assertTrue((result["hour_shifted"] <= 1).all())

    def test_all_features_in_present(self):
        from patchtst_model import FEATURES_IN
        result = self.add_features(self.df.copy())
        for feat in FEATURES_IN:
            self.assertIn(feat, result.columns, f"Brak cechy: {feat}")

    def test_no_nan_in_features(self):
        from patchtst_model import FEATURES_IN
        result = self.add_features(self.df.copy())
        nan_counts = result[FEATURES_IN].isna().sum()
        self.assertEqual(nan_counts.sum(), 0, f"NaN w cechach: {nan_counts[nan_counts > 0]}")


class TestPatchTSTModel(unittest.TestCase):
    """Testy architektury modelu PatchTST."""

    def test_build_model_returns_model(self):
        from patchtst_model import build_model, FEATURES_IN
        model = build_model(len(FEATURES_IN))
        self.assertIsNotNone(model)

    def test_model_parameter_count(self):
        from patchtst_model import build_model, FEATURES_IN
        model = build_model(len(FEATURES_IN))
        params = sum(p.numel() for p in model.parameters())
        # Sprawdź że model ma rozsądną liczbę parametrów (10k - 5M)
        self.assertGreater(params, 10_000)
        self.assertLess(params, 5_000_000)

    def test_model_forward_shape(self):
        import torch
        from patchtst_model import build_model, FEATURES_IN, INPUT_HOURS, OUTPUT_HOURS
        model = build_model(len(FEATURES_IN))
        model.eval()
        x = torch.randn(1, INPUT_HOURS, len(FEATURES_IN))
        with torch.no_grad():
            out = model(past_values=x)
        pred = out.prediction_outputs
        self.assertEqual(pred.shape[0], 1)           # batch
        self.assertEqual(pred.shape[1], OUTPUT_HOURS) # 24h
        self.assertEqual(pred.shape[2], len(FEATURES_IN))

    def test_model_forward_no_nan(self):
        import torch
        from patchtst_model import build_model, FEATURES_IN, INPUT_HOURS
        model = build_model(len(FEATURES_IN))
        model.eval()
        x = torch.randn(1, INPUT_HOURS, len(FEATURES_IN))
        with torch.no_grad():
            out = model(past_values=x)
        self.assertFalse(torch.isnan(out.prediction_outputs).any())


class TestPatchTSTClean(unittest.TestCase):
    """Testy funkcji _clean w patchtst_model."""

    def test_clean_fills_nan(self):
        from patchtst_model import _clean
        df = make_weather_df(50)
        df.loc[5:10, "temperature"] = np.nan
        result = _clean(df)
        self.assertEqual(result["temperature"].isna().sum(), 0)

    def test_clean_preserves_length(self):
        from patchtst_model import _clean
        df = make_weather_df(100)
        result = _clean(df)
        self.assertEqual(len(result), 100)


# ═══════════════════════════════════════════════════════
# TESTY: fish_model.py
# ═══════════════════════════════════════════════════════

class TestFishAddFeatures(unittest.TestCase):
    """Testy inżynierii cech dla modelu ryb."""

    def setUp(self):
        from fish_model import add_features
        self.add_features = add_features
        self.df = make_fish_df(100)

    def test_returns_dataframe(self):
        result = self.add_features(self.df.copy())
        self.assertIsInstance(result, pd.DataFrame)

    def test_pressure_diff_added(self):
        result = self.add_features(self.df.copy())
        self.assertIn("d_pressure_3h",  result.columns)
        self.assertIn("d_pressure_24h", result.columns)

    def test_rain_24h_sum_non_negative(self):
        result = self.add_features(self.df.copy())
        self.assertTrue((result["rain_24h_sum"] >= 0).all())

    def test_water_temp_computed(self):
        result = self.add_features(self.df.copy())
        self.assertIn("water_temp_c", result.columns)
        self.assertFalse(result["water_temp_c"].isna().all())

    def test_water_temp_delta_added(self):
        result = self.add_features(self.df.copy())
        self.assertIn("water_temp_delta", result.columns)

    def test_moon_phase_range(self):
        result = self.add_features(self.df.copy())
        self.assertTrue((result["moon_phase"] >= 0).all())
        self.assertTrue((result["moon_phase"] <= 1).all())

    def test_moon_illumination_range(self):
        result = self.add_features(self.df.copy())
        self.assertTrue((result["moon_illumination"] >= 0).all())
        self.assertTrue((result["moon_illumination"] <= 100).all())

    def test_is_dawn_binary(self):
        result = self.add_features(self.df.copy())
        self.assertTrue(result["is_dawn"].isin([0, 1]).all())

    def test_is_dusk_binary(self):
        result = self.add_features(self.df.copy())
        self.assertTrue(result["is_dusk"].isin([0, 1]).all())

    def test_all_feature_cols_present(self):
        from fish_model import FEATURE_COLS
        result = self.add_features(self.df.copy())
        for col in FEATURE_COLS:
            self.assertIn(col, result.columns, f"Brak kolumny: {col}")


class TestBiteProbability(unittest.TestCase):
    """Testy funkcji compute_bite_probability."""

    def setUp(self):
        from fish_model import add_features, compute_bite_probability, SPECIES
        self.compute = compute_bite_probability
        self.species = SPECIES
        df = make_fish_df(100)
        self.df = add_features(df)

    def test_returns_float(self):
        row = self.df.iloc[10]
        result = self.compute(row, "szczupak")
        self.assertIsInstance(result, float)

    def test_range_0_100(self):
        for species in self.species:
            for i in [0, 10, 50, 90]:
                row = self.df.iloc[i]
                prob = self.compute(row, species)
                self.assertGreaterEqual(prob, 0.0,   f"{species} prob < 0")
                self.assertLessEqual(prob,   100.0,  f"{species} prob > 100")

    def test_extreme_cold_returns_low(self):
        """Przy bardzo niskiej temperaturze wody ryby powinny być nieaktywne."""
        from fish_model import add_features
        df = make_fish_df(10)
        df["temperature_c"] = -5.0   # zimno
        df = add_features(df)
        row = df.iloc[5]
        prob = self.compute(row, "karp")
        self.assertLess(prob, 10.0, "Karp przy -5°C powinien miec <10%")

    def test_all_species_computed(self):
        row = self.df.iloc[20]
        for species in self.species:
            prob = self.compute(row, species)
            self.assertIsNotNone(prob)

    def test_sum_active_at_night(self):
        """Sum powinien miec wyższy score w nocy (20-6h)."""
        from fish_model import add_features
        # Dzień
        df_day = make_fish_df(5, start="2026-06-15 12:00")
        df_day = add_features(df_day)
        # Noc
        df_night = make_fish_df(5, start="2026-06-15 22:00")
        df_night = add_features(df_night)

        scores_day   = [self.compute(df_day.iloc[i],   "sum") for i in range(3)]
        scores_night = [self.compute(df_night.iloc[i], "sum") for i in range(3)]
        # Średnia nocna powinna być wyższa od dziennej
        self.assertGreater(np.mean(scores_night), np.mean(scores_day))


class TestMoonPhase(unittest.TestCase):
    """Testy funkcji moon_phase."""

    def test_returns_tuple(self):
        from fish_model import moon_phase
        result = moon_phase(pd.Timestamp("2026-01-01"))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_phase_range(self):
        from fish_model import moon_phase
        phase, illum = moon_phase(pd.Timestamp("2026-06-15"))
        self.assertGreaterEqual(phase, 0.0)
        self.assertLess(phase, 1.0)

    def test_illumination_range(self):
        from fish_model import moon_phase
        phase, illum = moon_phase(pd.Timestamp("2026-06-15"))
        self.assertGreaterEqual(illum, 0.0)
        self.assertLessEqual(illum, 100.0)

    def test_known_new_moon(self):
        from fish_model import moon_phase
        # 6 stycznia 2000 = znany nów
        phase, illum = moon_phase(pd.Timestamp("2000-01-06"))
        self.assertAlmostEqual(phase, 0.0, places=2)
        self.assertAlmostEqual(illum, 0.0, places=0)


class TestSolarElevation(unittest.TestCase):
    """Testy funkcji solar_elevation."""

    def test_returns_float(self):
        from fish_model import solar_elevation
        result = solar_elevation(pd.Timestamp("2026-06-21 12:00"), 53.0)
        self.assertIsInstance(result, float)

    def test_noon_higher_than_midnight(self):
        from fish_model import solar_elevation
        noon     = solar_elevation(pd.Timestamp("2026-06-21 12:00"), 53.0)
        midnight = solar_elevation(pd.Timestamp("2026-06-21 00:00"), 53.0)
        self.assertGreater(noon, midnight)

    def test_midnight_negative_elevation(self):
        from fish_model import solar_elevation
        elev = solar_elevation(pd.Timestamp("2026-06-21 00:00"), 53.0)
        self.assertLess(elev, 0)


class TestFitzroyLoader(unittest.TestCase):
    """Testy ładowania danych Fitzroy Basin."""

    def test_load_fitzroy_features(self):
        from fish_model import load_fitzroy_features, FEATURE_COLS
        csv_path = os.path.join(os.path.dirname(__file__), "Fitzroy_Basin.csv")
        if not os.path.exists(csv_path):
            self.skipTest("Fitzroy_Basin.csv niedostępny")

        df = load_fitzroy_features(csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 10)

        for col in FEATURE_COLS:
            self.assertIn(col, df.columns, f"Brak: {col}")

    def test_fitzroy_bite_probability_range(self):
        from fish_model import load_fitzroy_features
        csv_path = os.path.join(os.path.dirname(__file__), "Fitzroy_Basin.csv")
        if not os.path.exists(csv_path):
            self.skipTest("Fitzroy_Basin.csv niedostępny")

        df = load_fitzroy_features(csv_path)
        self.assertTrue((df["bite_probability"] >= 0).all())
        self.assertTrue((df["bite_probability"] <= 100).all())


# ═══════════════════════════════════════════════════════
# TESTY: weather_model.py (legacy TCN)
# ═══════════════════════════════════════════════════════

class TestWeatherModelFeatures(unittest.TestCase):
    """Testy add_features w weather_model."""

    def test_add_features_basic(self):
        from weather_model import add_features
        df = make_weather_df(100)
        result = add_features(df.copy())
        self.assertIn("hour_sin",       result.columns)
        self.assertIn("solar_heating",  result.columns)
        self.assertIn("hour_shifted",   result.columns)
        self.assertIn("temp_daily_max", result.columns)

    def test_fix_influx_cols_adds_wind_dir(self):
        from patchtst_model import _fix_influx_cols
        df = make_weather_df(20)
        df["wind_direction"] = np.random.uniform(0, 360, 20)
        result = _fix_influx_cols(df)
        self.assertIn("wind_dir_sin", result.columns)
        self.assertIn("wind_dir_cos", result.columns)

    def test_fix_influx_cols_adds_dewpoint(self):
        from patchtst_model import _fix_influx_cols
        df = make_weather_df(20)
        result = _fix_influx_cols(df)
        self.assertIsInstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════
# TESTY: tft_model.py
# ═══════════════════════════════════════════════════════

class TestTFTModelFeatures(unittest.TestCase):
    """Testy funkcji add_features i compute_pop w tft_model."""

    def test_compute_pop_range(self):
        from tft_model import compute_pop
        # Suche warunki
        pop_dry = compute_pop({
            "precip_sum": 0.0, "humidity_mean": 50,
            "cloudcover_mean": 20, "nwp_precip": 0.0
        })
        self.assertGreaterEqual(pop_dry, 0)
        self.assertLessEqual(pop_dry, 100)

        # Mokre warunki
        pop_wet = compute_pop({
            "precip_sum": 10.0, "humidity_mean": 95,
            "cloudcover_mean": 100, "nwp_precip": 8.0
        })
        self.assertGreater(pop_wet, pop_dry)
        self.assertLessEqual(pop_wet, 100)

    def test_compute_pop_returns_float(self):
        from tft_model import compute_pop
        result = compute_pop({
            "precip_sum": 2.0, "humidity_mean": 70,
            "cloudcover_mean": 60, "nwp_precip": 1.5
        })
        self.assertIsInstance(result, (int, float))


# ═══════════════════════════════════════════════════════
# TESTY: influx_writer.py
# ═══════════════════════════════════════════════════════

class TestInfluxWriter(unittest.TestCase):
    """Testy parsowania w influx_writer (bez połączenia z bazą)."""

    def _import_or_skip(self):
        try:
            import influx_writer
            return influx_writer
        except ImportError:
            self.skipTest("influx_writer niedostępny w PATH")

    def test_parse_ts_utc(self):
        m = self._import_or_skip()
        ts = m._parse_ts("2026-04-07T12:00:00")
        self.assertEqual(ts.hour, 12)

    def test_parse_ts_with_z(self):
        m = self._import_or_skip()
        ts = m._parse_ts("2026-04-07T12:00:00Z")
        self.assertIsNotNone(ts)

    def test_write_weather_24h_no_connection(self):
        """Zapis do InfluxDB gdy baza niedostępna powinien zwrócić 0 bez wyjątku."""
        m = self._import_or_skip()
        result = {
            "generated_at": "2026-04-07T12:00:00",
            "forecast": [
                {"timestamp": "2026-04-07T13:00:00", "temperature": 10.0}
            ]
        }
        with patch("influx_writer.InfluxDBClient") as mock_client:
            mock_client.side_effect = Exception("Connection refused")
            n = m.write_weather_24h(result)
        self.assertEqual(n, 0)


# ═══════════════════════════════════════════════════════
# TESTY: model_swap.py
# ═══════════════════════════════════════════════════════

class TestTrainingContext(unittest.TestCase):
    """Testy atomowej podmiany modelu."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.tmpdir, "model_files")
        os.makedirs(self.model_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_tmp_dir(self):
        from model_swap import TrainingContext
        ctx = TrainingContext(self.model_dir)
        ctx.__enter__()
        self.assertTrue(os.path.exists(ctx.tmp_dir))
        ctx.__exit__(None, None, None)

    def test_success_renames_tmp_to_active(self):
        from model_swap import TrainingContext
        ctx = TrainingContext(self.model_dir)
        ctx.__enter__()
        # Zapisz plik do tmp
        test_file = os.path.join(ctx.tmp_dir, "model.pth")
        with open(test_file, "w") as f:
            f.write("test")
        ctx.__exit__(None, None, None)
        # Po sukcesie plik powinien być w aktywnym katalogu
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "model.pth")))

    def test_failure_keeps_old_model(self):
        from model_swap import TrainingContext
        # Zapisz "stary" model
        old_file = os.path.join(self.model_dir, "old_model.pth")
        with open(old_file, "w") as f:
            f.write("old")

        ctx = TrainingContext(self.model_dir)
        ctx.__enter__()
        # Symuluj błąd treningu
        ctx.__exit__(ValueError, ValueError("trening się posypał"), None)
        # Stary model powinien pozostać
        self.assertTrue(os.path.exists(old_file))


# ═══════════════════════════════════════════════════════
# TESTY: openmeteo_to_influx.py
# ═══════════════════════════════════════════════════════

class TestOpenMeteoToInflux(unittest.TestCase):
    """Testy funkcji pomocniczych w openmeteo_to_influx."""

    def test_estimate_water_sensors_returns_df(self):
        from openmeteo_to_influx import estimate_water_sensors
        df = pd.DataFrame({
            "timestamp":      pd.date_range("2026-01-01", periods=50, freq="h"),
            "temperature_c":  np.random.uniform(0, 20, 50),
            "humidity_pct":   np.random.uniform(50, 90, 50),
            "pressure_hpa":   np.random.uniform(995, 1025, 50),
            "wind_speed_kmh": np.random.uniform(0, 20, 50),
            "wind_gust_kmh":  np.random.uniform(0, 30, 50),
            "rain_rate_mm":   np.random.uniform(0, 2, 50),
            "cloudcover_pct": np.random.uniform(0, 100, 50),
            "shortwave_rad":  np.random.uniform(0, 500, 50),
        })
        result = estimate_water_sensors(df)
        self.assertIn("water_temp_c",    result.columns)
        self.assertIn("water_level_cm",  result.columns)
        self.assertIn("water_ph",        result.columns)
        self.assertIn("water_oxygen",    result.columns)
        self.assertIn("turbidity_ntu",   result.columns)

    def _make_df(self, n=200):
        """DataFrame z wystarczajaca historia dla rolling(72h)."""
        return pd.DataFrame({
            "timestamp":        pd.date_range("2026-01-01", periods=n, freq="h"),
            "temperature_c":    [15.0] * n,
            "humidity_pct":     [70.0] * n,
            "pressure_hpa":     [1013.0] * n,
            "wind_speed_kmh":   [5.0] * n,
            "wind_gust_kmh":    [10.0] * n,
            "rain_rate_mm":     [0.0] * n,
            "cloudcover_pct":   [50.0] * n,
            "shortwave_rad":    [200.0] * n,
        })

    def test_water_ph_range(self):
        from openmeteo_to_influx import estimate_water_sensors
        result = estimate_water_sensors(self._make_df(200))
        # Sprawdz tylko wiersze po rozgrzewce rolling(72h)
        result = result.iloc[80:]
        self.assertEqual(result["water_ph"].isna().sum(), 0)
        self.assertTrue((result["water_ph"] >= 5.0).all())
        self.assertTrue((result["water_ph"] <= 10.0).all())

    def test_water_oxygen_positive(self):
        from openmeteo_to_influx import estimate_water_sensors
        result = estimate_water_sensors(self._make_df(200))
        result = result.iloc[80:]
        self.assertEqual(result["water_oxygen"].isna().sum(), 0)
        self.assertGreater(result["water_oxygen"].mean(), 0)


# ═══════════════════════════════════════════════════════
# PYTEST FIXTURES I PARAMETRYCZNE
# ═══════════════════════════════════════════════════════

@pytest.fixture
def weather_df():
    return make_weather_df(200)


@pytest.fixture
def fish_df():
    return make_fish_df(100)


@pytest.mark.parametrize("species", [
    "karp", "leszcz", "szczupak", "okon", "sandacz",
    "ploc", "lin", "sum", "klen", "wzdrega"
])
def test_bite_probability_all_species(species, fish_df):
    """Każdy gatunek powinien zwracać wartość 0-100."""
    from fish_model import add_features, compute_bite_probability
    df = add_features(fish_df)
    for i in range(0, min(10, len(df))):
        prob = compute_bite_probability(df.iloc[i], species)
        assert 0 <= prob <= 100, f"{species}: prob={prob} poza zakresem"


@pytest.mark.parametrize("hours,expected_min_rows", [
    (168, 168),
    (200, 200),
    (500, 500),
])
def test_patchtst_features_length(hours, expected_min_rows):
    """add_features nie powinno zmieniać liczby wierszy."""
    from patchtst_model import add_features
    df = make_weather_df(hours)
    result = add_features(df)
    assert len(result) >= expected_min_rows


@pytest.mark.parametrize("temp,species,expected_max_prob", [
    (-10, "karp",    10),   # karp poniżej minimum
    (-10, "leszcz",  10),   # leszcz poniżej minimum
    (35,  "szczupak", 15),  # szczupak powyżej maximum
    (40,  "wzdrega",  10),  # wzdręga powyżej maximum
])
def test_extreme_temperatures(temp, species, expected_max_prob, fish_df):
    """Ekstremalne temperatury powinny dawać niskie prawdopodobieństwo."""
    from fish_model import add_features, compute_bite_probability
    fish_df = fish_df.copy()
    fish_df["temperature_c"] = float(temp)
    df = add_features(fish_df)
    probs = [compute_bite_probability(df.iloc[i], species) for i in range(5)]
    avg = np.mean(probs)
    assert avg <= expected_max_prob, \
        f"{species} przy {temp}°C: średnia={avg:.1f}% > {expected_max_prob}%"


# ─────────────────────────────────────────────
# URUCHOMIENIE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
