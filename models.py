"""
models.py – Unified ML engine for the merged F1 project.

Contains:
  - RaceLSTM   : PyTorch LSTM model for sequence prediction
  - ModelEngine : Feature engineering, XGBoost training, autoregressive
                  pace prediction, win probability, telemetry interpolation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# ===========================================================================
#  PyTorch LSTM
# ===========================================================================

class RaceLSTM(nn.Module):
    """LSTM for lap-level race sequence prediction.

    Parameters
    ----------
    input_size : int
        Number of features per time step (default 10).
    hidden_size : int
        LSTM hidden dimension (default 64).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    output_size : int
        Prediction output dimension (default 1 – e.g. lap time).
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass. *x* shape: ``(batch, seq_len, input_size)``."""
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # last time-step
        return out


# ===========================================================================
#  Model Engine
# ===========================================================================

class ModelEngine:
    """Unified ML toolkit: feature engineering, XGBoost, predictions, probabilities."""

    # -----------------------------------------------------------------------
    # Telemetry interpolation
    # -----------------------------------------------------------------------

    @staticmethod
    def interpolate_telemetry(
        telemetry: pd.DataFrame,
        num_points: int = 1000,
    ) -> pd.DataFrame:
        """Spatially interpolate telemetry to *num_points* evenly-spaced distances.

        Expects a ``Distance`` column and returns ``Speed``, ``Throttle``, ``Brake``
        (plus any other numeric columns) at uniform distance intervals.
        """
        if telemetry.empty or "Distance" not in telemetry.columns:
            return telemetry

        df = telemetry.sort_values("Distance").copy()
        dist = df["Distance"].values
        if len(dist) < 2:
            return df

        new_dist = np.linspace(dist.min(), dist.max(), num_points)
        result: Dict[str, np.ndarray] = {"Distance": new_dist}

        for col in df.columns:
            if col == "Distance":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    f_interp = interp1d(dist, df[col].values, kind="linear", fill_value="extrapolate")
                    result[col] = f_interp(new_dist)
                except Exception:
                    continue

        return pd.DataFrame(result)

    # -----------------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------------

    @staticmethod
    def prepare_pace_features(laps: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering for pace prediction.

        Input DataFrame should contain at minimum:
            LapNumber, LapTime (timedelta or float seconds), TyreLife, Compound, Stint

        Added features:
            LapTime_sec, Prev_LapTime_sec (autoregressive), Compound_enc, TyreLife,
            Stint, LapNumber, FuelLoad (estimated linear decrease),
            AvgThrottle, AvgBrake, IsSC
        """
        df = laps.copy()

        # LapTime → seconds
        if "LapTime_sec" not in df.columns:
            if "LapTime" in df.columns:
                if pd.api.types.is_timedelta64_any_dtype(df["LapTime"]):
                    df["LapTime_sec"] = df["LapTime"].dt.total_seconds()
                else:
                    df["LapTime_sec"] = pd.to_numeric(df["LapTime"], errors="coerce")
            else:
                df["LapTime_sec"] = np.nan

        # Remove rows with no lap time
        df = df.dropna(subset=["LapTime_sec"]).copy()

        # Autoregressive lag feature
        df["Prev_LapTime_sec"] = df["LapTime_sec"].shift(1)
        df["Prev_LapTime_sec"] = df["Prev_LapTime_sec"].bfill()

        # Compound encoding
        compound_map = {
            "SOFT": 0,
            "MEDIUM": 1,
            "HARD": 2,
            "INTERMEDIATE": 3,
            "WET": 4,
            "UNKNOWN": 1,
        }
        if "Compound" in df.columns:
            df["Compound_enc"] = (
                df["Compound"].str.upper().map(compound_map).fillna(1).astype(int)
            )
        else:
            df["Compound_enc"] = 1

        # TyreLife default
        if "TyreLife" not in df.columns:
            df["TyreLife"] = 1

        # Stint default
        if "Stint" not in df.columns:
            df["Stint"] = 1

        # LapNumber default
        if "LapNumber" not in df.columns:
            df["LapNumber"] = range(1, len(df) + 1)

        # Fuel load estimate (linear decrease, starting from ~110 kg)
        max_lap = df["LapNumber"].max()
        if max_lap > 0:
            df["FuelLoad"] = 110.0 * (1.0 - df["LapNumber"] / max_lap)
        else:
            df["FuelLoad"] = 110.0

        # Fill optional columns
        for col in ("AvgThrottle", "AvgBrake", "IsSC"):
            if col not in df.columns:
                df[col] = 0.0

        return df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # XGBoost training
    # -----------------------------------------------------------------------

    FEATURE_COLS: List[str] = [
        "LapNumber",
        "TyreLife",
        "Compound_enc",
        "Stint",
        "FuelLoad",
        "Prev_LapTime_sec",
        "AvgThrottle",
        "AvgBrake",
        "IsSC",
    ]

    @staticmethod
    def train_pace_model(
        df: pd.DataFrame,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
    ) -> xgb.XGBRegressor:
        """Train an XGBoost regressor on prepared lap data.

        Parameters
        ----------
        df : DataFrame
            Output of :meth:`prepare_pace_features`.
        n_estimators : int
            Number of boosting rounds.
        learning_rate : float
            Step size shrinkage.

        Returns
        -------
        xgb.XGBRegressor
            Fitted model.
        """
        feature_cols = [c for c in ModelEngine.FEATURE_COLS if c in df.columns]
        target = "LapTime_sec"

        X = df[feature_cols].copy()
        y = df[target].copy()

        # Drop any remaining NaN rows
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            verbosity=0,
        )
        model.fit(X, y)
        return model

    # -----------------------------------------------------------------------
    # Autoregressive prediction
    # -----------------------------------------------------------------------

    @staticmethod
    def predict_future_pace(
        model: xgb.XGBRegressor,
        last_known: pd.Series,
        laps_ahead: int = 10,
        compound: str = "MEDIUM",
        stint: int = 2,
        tyre_life_start: int = 1,
    ) -> pd.DataFrame:
        """Autoregressively predict future lap times.

        Parameters
        ----------
        model : XGBRegressor
            Fitted pace model.
        last_known : Series
            Feature row of the last observed lap.
        laps_ahead : int
            Number of laps to predict.
        compound : str
            Compound for future stints.
        stint : int
            Stint number for future laps.
        tyre_life_start : int
            Tyre life at the start of prediction window.

        Returns
        -------
        DataFrame
            Predicted laps with all feature columns + ``Predicted_LapTime_sec``.
        """
        compound_map = {
            "SOFT": 0,
            "MEDIUM": 1,
            "HARD": 2,
            "INTERMEDIATE": 3,
            "WET": 4,
        }
        compound_enc = compound_map.get(compound.upper(), 1)

        feature_cols = [c for c in ModelEngine.FEATURE_COLS if c in last_known.index]
        predictions: List[Dict[str, Any]] = []
        prev_time = float(last_known.get("LapTime_sec", last_known.get("Prev_LapTime_sec", 90)))

        start_lap = int(last_known.get("LapNumber", 0)) + 1
        max_lap = start_lap + laps_ahead

        for i in range(laps_ahead):
            lap_num = start_lap + i
            tyre_life = tyre_life_start + i
            fuel_load = 110.0 * (1.0 - lap_num / max_lap) if max_lap > 0 else 0

            row: Dict[str, float] = {
                "LapNumber": lap_num,
                "TyreLife": tyre_life,
                "Compound_enc": compound_enc,
                "Stint": stint,
                "FuelLoad": fuel_load,
                "Prev_LapTime_sec": prev_time,
                "AvgThrottle": float(last_known.get("AvgThrottle", 0)),
                "AvgBrake": float(last_known.get("AvgBrake", 0)),
                "IsSC": 0,
            }

            X_pred = pd.DataFrame([row])[feature_cols]
            pred_time = float(model.predict(X_pred)[0])
            row["Predicted_LapTime_sec"] = pred_time
            predictions.append(row)
            prev_time = pred_time

        return pd.DataFrame(predictions)

    # -----------------------------------------------------------------------
    # Win probability (position-based, no dummy calculations)
    # -----------------------------------------------------------------------

    @staticmethod
    def get_real_win_probability(
        lap_metrics_df: pd.DataFrame,
        pace_models: Dict[str, xgb.XGBRegressor],
        remaining_laps: int = 10,
    ) -> Dict[str, float]:
        """Estimate win probability for each driver based on current positions
        and predicted pace.

        Parameters
        ----------
        lap_metrics_df : DataFrame
            Latest lap metrics with at least ``Driver`` and ``Position`` columns,
            optionally ``LapTime_sec`` or ``LapTime``.
        pace_models : dict
            ``{driver_abbreviation: fitted_XGBRegressor}``.
        remaining_laps : int
            Number of laps remaining in the race.

        Returns
        -------
        dict
            ``{driver: probability}`` summing to 1.0.
        """
        if lap_metrics_df.empty or "Driver" not in lap_metrics_df.columns:
            return {}

        scores: Dict[str, float] = {}

        for _, row in lap_metrics_df.iterrows():
            driver = row["Driver"]
            position = row.get("Position", 20)
            try:
                position = float(position)
            except (ValueError, TypeError):
                position = 20.0

            # Base score from position (lower position = higher score)
            base_score = max(0.0, 21.0 - position)

            # Bonus from pace model if available
            if driver in pace_models:
                try:
                    model = pace_models[driver]
                    # Use current lap data as base for prediction
                    pred_row = row.copy()
                    if "LapTime_sec" not in pred_row or pd.isna(pred_row.get("LapTime_sec")):
                        if "LapTime" in pred_row:
                            lt = pred_row["LapTime"]
                            if hasattr(lt, "total_seconds"):
                                pred_row["LapTime_sec"] = lt.total_seconds()
                    pred = ModelEngine.predict_future_pace(
                        model,
                        pred_row,
                        laps_ahead=min(remaining_laps, 5),
                    )
                    if not pred.empty:
                        avg_predicted = pred["Predicted_LapTime_sec"].mean()
                        # Faster predicted pace → higher bonus
                        pace_bonus = max(0.0, 100.0 - avg_predicted) * 0.1
                        base_score += pace_bonus
                except Exception:
                    pass

            scores[driver] = max(base_score, 0.01)

        # Normalise to probabilities
        total = sum(scores.values())
        if total > 0:
            return {d: s / total for d, s in scores.items()}
        return {d: 1.0 / len(scores) for d in scores}
