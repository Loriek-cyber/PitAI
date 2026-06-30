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
            "Throttle": [50, 70, 90, 100, 95, 80],
        })
        result = me.interpolate_telemetry(df, num_points=10)
        assert len(result) == 10
        assert "Distance" in result.columns
        assert "Speed" in result.columns
        assert "Throttle" in result.columns

    def test_empty_dataframe(self):
        me = ModelEngine()
        df = pd.DataFrame()
        result = me.interpolate_telemetry(df)
        assert result.empty

    def test_missing_distance_column(self):
        me = ModelEngine()
        df = pd.DataFrame({"Speed": [100, 200]})
        result = me.interpolate_telemetry(df)
        assert len(result) == 2  # returned unchanged

    def test_single_row(self):
        me = ModelEngine()
        df = pd.DataFrame({"Distance": [100], "Speed": [200]})
        result = me.interpolate_telemetry(df, num_points=10)
        assert len(result) == 1  # too few points to interpolate


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestPreparePaceFeatures:
    def test_basic_features(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": [1, 2, 3, 4, 5],
            "LapTime": pd.to_timedelta([90, 91, 89, 92, 88], unit="s"),
            "TyreLife": [1, 2, 3, 4, 5],
            "Compound": ["SOFT", "SOFT", "SOFT", "SOFT", "SOFT"],
            "Stint": [1, 1, 1, 1, 1],
        })
        result = me.prepare_pace_features(laps)

        assert "LapTime_sec" in result.columns
        assert "Prev_LapTime_sec" in result.columns
        assert "Compound_enc" in result.columns
        assert "FuelLoad" in result.columns
        assert len(result) == 5

    def test_compound_encoding(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapTime": pd.to_timedelta([90, 91, 89], unit="s"),
            "Compound": ["SOFT", "MEDIUM", "HARD"],
        })
        result = me.prepare_pace_features(laps)
        assert result["Compound_enc"].tolist() == [0, 1, 2]

    def test_handles_missing_columns(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapTime": pd.to_timedelta([90, 91], unit="s"),
        })
        result = me.prepare_pace_features(laps)
        assert "TyreLife" in result.columns
        assert "Stint" in result.columns
        assert "Compound_enc" in result.columns

    def test_autoregressive_feature(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapTime": pd.to_timedelta([90, 91, 89], unit="s"),
        })
        result = me.prepare_pace_features(laps)
        # First row should be backfilled from itself
        assert result["Prev_LapTime_sec"].iloc[1] == 90.0

    def test_fuel_load_decreases(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": list(range(1, 11)),
            "LapTime": pd.to_timedelta([90] * 10, unit="s"),
        })
        result = me.prepare_pace_features(laps)
        assert result["FuelLoad"].iloc[0] > result["FuelLoad"].iloc[-1]

    def test_drops_na_lap_times(self):
        me = ModelEngine()
        laps = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapTime": [pd.NaT, pd.to_timedelta(90, unit="s"), pd.to_timedelta(91, unit="s")],
        })
        result = me.prepare_pace_features(laps)
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
        return me.prepare_pace_features(laps)

    def test_model_trains(self):
        me = ModelEngine()
        df = self._make_training_data()
        model = me.train_pace_model(df)
        assert model is not None

    def test_model_predicts(self):
        me = ModelEngine()
        df = self._make_training_data()
        model = me.train_pace_model(df)
        preds = model.predict(df[ModelEngine.FEATURE_COLS])
        assert len(preds) == len(df)
        assert all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Autoregressive prediction
# ---------------------------------------------------------------------------

class TestPredictFuturePace:
    def test_prediction_count(self):
        me = ModelEngine()
        # Build small training set
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
        df = me.prepare_pace_features(laps)
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
        df = me.prepare_pace_features(laps)
        model = me.train_pace_model(df)
        last = df.iloc[-1]
        preds = me.predict_future_pace(model, last, laps_ahead=5, compound="HARD")
        assert all(np.isfinite(preds["Predicted_LapTime_sec"]))


# ---------------------------------------------------------------------------
# Win probability
# ---------------------------------------------------------------------------

class TestGetRealWinProbability:
    def test_probabilities_sum_to_one(self):
        me = ModelEngine()
        metrics = pd.DataFrame({
            "Driver": ["VER", "HAM", "LEC"],
            "Position": [1, 2, 3],
        })
        probs = me.get_real_win_probability(metrics, {})
        assert pytest.approx(sum(probs.values()), abs=1e-6) == 1.0

    def test_leader_has_highest_probability(self):
        me = ModelEngine()
        metrics = pd.DataFrame({
            "Driver": ["VER", "HAM", "LEC"],
            "Position": [1, 2, 3],
        })
        probs = me.get_real_win_probability(metrics, {})
        assert probs["VER"] > probs["HAM"]
        assert probs["HAM"] > probs["LEC"]

    def test_empty_input(self):
        me = ModelEngine()
        probs = me.get_real_win_probability(pd.DataFrame(), {})
        assert probs == {}

    def test_all_drivers_present(self):
        me = ModelEngine()
        metrics = pd.DataFrame({
            "Driver": ["VER", "HAM", "LEC", "NOR", "PIA"],
            "Position": [1, 2, 3, 4, 5],
        })
        probs = me.get_real_win_probability(metrics, {})
        assert set(probs.keys()) == {"VER", "HAM", "LEC", "NOR", "PIA"}
