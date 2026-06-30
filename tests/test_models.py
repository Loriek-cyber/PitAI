"""
test_models.py – Pytest tests for the ModelEngine and RaceLSTM.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import ModelEngine, RaceLSTM


# ---------------------------------------------------------------------------
# RaceLSTM
# ---------------------------------------------------------------------------

class TestRaceLSTM:
    def test_output_shape(self):
        model = RaceLSTM(input_size=10, hidden_size=64, num_layers=2, output_size=1)
        x = torch.randn(4, 5, 10)  # batch=4, seq_len=5, features=10
        out = model(x)
        assert out.shape == (4, 1)

    def test_single_sample(self):
        model = RaceLSTM(input_size=10, hidden_size=64, num_layers=2, output_size=1)
        x = torch.randn(1, 1, 10)
        out = model(x)
        assert out.shape == (1, 1)
        assert not torch.isnan(out).any()

    def test_custom_output_size(self):
        model = RaceLSTM(input_size=5, hidden_size=32, num_layers=1, output_size=3)
        x = torch.randn(2, 10, 5)
        out = model(x)
        assert out.shape == (2, 3)

    def test_gradient_flow(self):
        model = RaceLSTM(input_size=10, hidden_size=64, num_layers=2, output_size=1)
        x = torch.randn(2, 5, 10, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Telemetry interpolation
# ---------------------------------------------------------------------------

class TestInterpolateTelemetry:
    def test_basic_interpolation(self):
        me = ModelEngine()
        df = pd.DataFrame({
            "Distance": [0, 100, 200, 300, 400, 500],
            "Speed": [100, 150, 200, 250, 300, 280],
        })
        dist, speed = me.interpolate_telemetry(df, num_points=10)
        assert len(dist) == 10
        assert len(speed) == 10
        # First and last values should match input endpoints
        assert dist[0] == 0
        assert dist[-1] == 500

    def test_returns_tuple(self):
        me = ModelEngine()
        df = pd.DataFrame({
            "Distance": [0, 100, 200],
            "Speed": [100, 200, 300],
        })
        result = me.interpolate_telemetry(df)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)

    def test_empty_dataframe(self):
        me = ModelEngine()
        df = pd.DataFrame()
        dist, speed = me.interpolate_telemetry(df)
        assert len(dist) == 1000
        assert len(speed) == 1000

    def test_missing_distance_column(self):
        me = ModelEngine()
        df = pd.DataFrame({"Speed": [100, 200]})
        dist, speed = me.interpolate_telemetry(df)
        # Returns placeholder arrays
        assert len(dist) == 1000

    def test_single_row(self):
        me = ModelEngine()
        df = pd.DataFrame({"Distance": [100], "Speed": [200]})
        dist, speed = me.interpolate_telemetry(df, num_points=10)
        assert len(dist) == 1
        assert len(speed) == 1


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestPreparePaceFeatures:
    def test_returns_tuple(self):
        """prepare_pace_features must return (DataFrame, LabelEncoder)."""
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapTime": pd.to_timedelta([90, 91, 89], unit="s"),
            "Compound": ["SOFT", "SOFT", "SOFT"],
        })
        result = me.prepare_pace_features(laps)
        assert isinstance(result, tuple)
        assert len(result) == 2
        df, le = result
        assert isinstance(df, pd.DataFrame)

    def test_basic_features(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": [1, 2, 3, 4, 5],
            "LapTime": pd.to_timedelta([90, 91, 89, 92, 88], unit="s"),
            "TyreLife": [1, 2, 3, 4, 5],
            "Compound": ["SOFT", "SOFT", "SOFT", "SOFT", "SOFT"],
            "Stint": [1, 1, 1, 1, 1],
        })
        result, le = me.prepare_pace_features(laps)

        assert "LapTime_sec" in result.columns
        assert "Prev_LapTime_sec" in result.columns
        assert "Compound_enc" in result.columns
        assert "Compound_encoded" in result.columns  # backward compat
        assert "FuelLoad" in result.columns
        assert len(result) == 5

    def test_compound_encoding_static_map(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapTime": pd.to_timedelta([90, 91, 89], unit="s"),
            "Compound": ["SOFT", "MEDIUM", "HARD"],
        })
        result, le = me.prepare_pace_features(laps)
        # Static map: SOFT=0, MEDIUM=1, HARD=2
        assert result["Compound_enc"].tolist() == [0, 1, 2]

    def test_handles_missing_columns(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapTime": pd.to_timedelta([90, 91], unit="s"),
        })
        result, le = me.prepare_pace_features(laps)
        assert "TyreLife" in result.columns
        assert "Stint" in result.columns
        assert "Compound_enc" in result.columns

    def test_autoregressive_feature(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapTime": pd.to_timedelta([90, 91, 89], unit="s"),
        })
        result, le = me.prepare_pace_features(laps)
        # First row should be backfilled from itself
        assert result["Prev_LapTime_sec"].iloc[1] == 90.0

    def test_fuel_load_decreases(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": list(range(1, 11)),
            "LapTime": pd.to_timedelta([90] * 10, unit="s"),
        })
        result, le = me.prepare_pace_features(laps)
        assert result["FuelLoad"].iloc[0] > result["FuelLoad"].iloc[-1]

    def test_drops_na_lap_times(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapTime": [pd.NaT, pd.to_timedelta(90, unit="s"), pd.to_timedelta(91, unit="s")],
        })
        result, le = me.prepare_pace_features(laps)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

class TestTrainPaceModel:
    def _make_training_data(self, n=50):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": list(range(1, n + 1)),
            "LapTime": pd.to_timedelta(np.random.uniform(85, 95, n), unit="s"),
            "TyreLife": list(range(1, n + 1)),
            "Compound": ["MEDIUM"] * n,
            "Stint": [1] * (n // 2) + [2] * (n - n // 2),
            "AvgThrottle": np.random.uniform(60, 90, n),
            "AvgBrake": np.random.uniform(5, 15, n),
            "IsSC": [0] * n,
        })
        df, le = me.prepare_pace_features(laps)
        return df

    def test_model_trains(self):
        me = ModelEngine()
        df = self._make_training_data()
        model = me.train_pace_model(df)
        assert model is not None

    def test_model_predicts(self):
        me = ModelEngine()
        df = self._make_training_data()
        model = me.train_pace_model(df)
        feature_cols = [c for c in ModelEngine.FEATURE_COLS if c in df.columns]
        preds = model.predict(df[feature_cols])
        assert len(preds) == len(df)
        assert all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Autoregressive prediction (Series-based)
# ---------------------------------------------------------------------------

class TestPredictFuturePace:
    def test_prediction_count(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": list(range(1, 31)),
            "LapTime": pd.to_timedelta(np.random.uniform(88, 93, 30), unit="s"),
            "TyreLife": list(range(1, 31)),
            "Compound": ["SOFT"] * 30,
            "Stint": [1] * 30,
            "AvgThrottle": [75.0] * 30,
            "AvgBrake": [10.0] * 30,
            "IsSC": [0] * 30,
        })
        df, le = me.prepare_pace_features(laps)
        model = me.train_pace_model(df)

        last = df.iloc[-1]
        preds = me.predict_future_pace(model, last, laps_ahead=10)
        assert len(preds) == 10
        assert "Predicted_LapTime_sec" in preds.columns

    def test_predictions_are_finite(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": list(range(1, 21)),
            "LapTime": pd.to_timedelta([90] * 20, unit="s"),
            "TyreLife": list(range(1, 21)),
            "Compound": ["HARD"] * 20,
            "Stint": [1] * 20,
        })
        df, le = me.prepare_pace_features(laps)
        model = me.train_pace_model(df)
        last = df.iloc[-1]
        preds = me.predict_future_pace(model, last, laps_ahead=5, compound="HARD")
        assert all(np.isfinite(preds["Predicted_LapTime_sec"]))


# ---------------------------------------------------------------------------
# Autoregressive prediction (scalar-based, app.py interface)
# ---------------------------------------------------------------------------

class TestPredictFuturePaceFromState:
    def test_prediction_count(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": list(range(1, 31)),
            "LapTime": pd.to_timedelta(np.random.uniform(88, 93, 30), unit="s"),
            "TyreLife": list(range(1, 31)),
            "Compound": ["SOFT"] * 30,
            "Stint": [1] * 30,
            "AvgThrottle": [75.0] * 30,
            "AvgBrake": [10.0] * 30,
            "IsSC": [0] * 30,
        })
        df, le = me.prepare_pace_features(laps)
        model = me.train_pace_model(df)

        preds = me.predict_future_pace_from_state(
            model,
            current_lap=30,
            current_tyre_life=30,
            current_compound_enc=0,  # SOFT
            current_stint=1,
            current_laptime=90.0,
            avg_throttle=75.0,
            avg_brake=10.0,
            num_laps=10,
        )
        assert len(preds) == 10
        assert "Predicted_LapTime_sec" in preds.columns
        assert "LapNumber" in preds.columns
        assert "TyreLife" in preds.columns

    def test_predictions_are_finite(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": list(range(1, 21)),
            "LapTime": pd.to_timedelta([90] * 20, unit="s"),
            "TyreLife": list(range(1, 21)),
            "Compound": ["MEDIUM"] * 20,
            "Stint": [1] * 20,
        })
        df, le = me.prepare_pace_features(laps)
        model = me.train_pace_model(df)

        preds = me.predict_future_pace_from_state(
            model, 20, 20, 1, 1, 90.0, 70.0, 10.0, num_laps=5
        )
        assert all(np.isfinite(preds["Predicted_LapTime_sec"]))

    def test_autoregression_effect(self):
        """Predicted lap times should be finite and in a reasonable range."""
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": list(range(1, 51)),
            "LapTime": pd.to_timedelta(np.linspace(88, 95, 50), unit="s"),
            "TyreLife": list(range(1, 51)),
            "Compound": ["SOFT"] * 50,
            "Stint": [1] * 50,
            "AvgThrottle": [75.0] * 50,
            "AvgBrake": [10.0] * 50,
            "IsSC": [0] * 50,
        })
        df, le = me.prepare_pace_features(laps)
        model = me.train_pace_model(df)

        preds = me.predict_future_pace_from_state(
            model, 50, 50, 0, 1, 95.0, 75.0, 10.0, num_laps=5
        )
        times = preds["Predicted_LapTime_sec"].tolist()
        # All predictions should be finite and within a reasonable range
        assert all(np.isfinite(times))
        assert all(60 < t < 200 for t in times), f"Predictions out of range: {times}"
        # Lap numbers should increment correctly
        assert preds["LapNumber"].tolist() == [51, 52, 53, 54, 55]


# ---------------------------------------------------------------------------
# Win probability
# ---------------------------------------------------------------------------

class TestGetRealWinProbability:
    def test_returns_list_of_dicts(self):
        me = ModelEngine()
        metrics = pd.DataFrame({
            "Driver": ["VER", "HAM", "LEC"],
            "Position": [1, 2, 3],
        })
        result = me.get_real_win_probability(metrics, {})
        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], dict)

    def test_probabilities_sum_to_one(self):
        me = ModelEngine()
        metrics = pd.DataFrame({
            "Driver": ["VER", "HAM", "LEC"],
            "Position": [1, 2, 3],
        })
        result = me.get_real_win_probability(metrics, {})
        total_prob = sum(r["Probabilità"] for r in result)
        assert pytest.approx(total_prob, abs=1e-6) == 1.0

    def test_leader_has_highest_probability(self):
        me = ModelEngine()
        metrics = pd.DataFrame({
            "Driver": ["VER", "HAM", "LEC"],
            "Position": [1, 2, 3],
        })
        result = me.get_real_win_probability(metrics, {})
        probs = {r["Pilota"]: r["Probabilità"] for r in result}
        assert probs["VER"] > probs["HAM"]
        assert probs["HAM"] > probs["LEC"]

    def test_empty_input(self):
        me = ModelEngine()
        result = me.get_real_win_probability(pd.DataFrame(), {})
        assert result == []

    def test_all_drivers_present(self):
        me = ModelEngine()
        metrics = pd.DataFrame({
            "Driver": ["VER", "HAM", "LEC", "NOR", "PIA"],
            "Position": [1, 2, 3, 4, 5],
        })
        result = me.get_real_win_probability(metrics, {})
        drivers = {r["Pilota"] for r in result}
        assert drivers == {"VER", "HAM", "LEC", "NOR", "PIA"}

    def test_rich_output_columns(self):
        """Verify the output includes all the columns needed by the Streamlit app."""
        me = ModelEngine()
        metrics = pd.DataFrame({
            "Driver": ["VER", "HAM"],
            "Position": [1, 2],
            "LapTime": pd.to_timedelta([90, 91], unit="s"),
            "Compound": ["SOFT", "MEDIUM"],
            "TyreLife": [10, 15],
            "SpeedST": [320, 315],
            "TrackStatus": ["1", "1"],
            "Stint": [1, 2],
            "LapNumber": [30, 30],
            "TotalLaps": [53, 53],
        })
        result = me.get_real_win_probability(metrics, {})
        expected_keys = {
            "Posizione", "Pilota", "Probabilità", "Andamento",
            "Finestra Pit", "Carburante", "Pit Stop", "Accelerazione",
            "Tempo", "Gomma", "Età Gomma", "Vel. Max", "Traffico (Stato)",
        }
        assert expected_keys == set(result[0].keys())

    def test_fuel_estimation(self):
        """Fuel should decrease as lap number increases."""
        me = ModelEngine()
        metrics = pd.DataFrame({
            "Driver": ["VER", "HAM"],
            "Position": [1, 2],
            "LapNumber": [10, 40],
            "TotalLaps": [53, 53],
        })
        result = me.get_real_win_probability(metrics, {})
        # VER at lap 10 should have more fuel than HAM at lap 40
        fuel_ver = float(result[0]["Carburante"].replace(" kg", ""))
        fuel_ham = float(result[1]["Carburante"].replace(" kg", ""))
        assert fuel_ver > fuel_ham
