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
from sklearn.preprocessing import LabelEncoder

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Spatially interpolate telemetry to *num_points* evenly-spaced distances.

        Returns a tuple ``(distance_array, speed_array)`` suitable for direct
        plotting in the Speed Trace tab.

        Expects ``Distance`` and ``Speed`` columns.  If either is missing or
        the frame is empty, returns placeholder zero-arrays.
        """
        if (
            telemetry.empty
            or "Distance" not in telemetry.columns
            or "Speed" not in telemetry.columns
        ):
            return np.linspace(0, 1000, num_points), np.zeros(num_points)

        tel_clean = telemetry.dropna(subset=["Distance", "Speed"]).copy()
        tel_clean = tel_clean.sort_values("Distance")

        dist = tel_clean["Distance"].values
        speed = tel_clean["Speed"].values

        if len(dist) < 2:
            return dist, speed

        common_distance = np.linspace(dist.min(), dist.max(), num_points)
        speed_interp = np.interp(common_distance, dist, speed)

        return common_distance, speed_interp

    # -----------------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------------

    # Static compound map used both in prepare and predict
    COMPOUND_MAP: Dict[str, int] = {
        "SOFT": 0,
        "MEDIUM": 1,
        "HARD": 2,
        "INTERMEDIATE": 3,
        "WET": 4,
        "UNKNOWN": 1,
    }

    @staticmethod
    def prepare_pace_features(
        laps: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, LabelEncoder]:
        """Full feature engineering for pace prediction.

        Input DataFrame should contain at minimum:
            LapNumber, LapTime (timedelta or float seconds), TyreLife, Compound, Stint

        Returns
        -------
        tuple[DataFrame, LabelEncoder]
            - The prepared DataFrame with all features ready for XGBoost.
            - A fitted LabelEncoder for the ``Compound`` column (also stored
              as ``Compound_encoded`` for backward compatibility with Rob's
              app code).

        Added features:
            LapTime_sec, Prev_LapTime_sec (autoregressive), Compound_enc,
            Compound_encoded (alias), TyreLife, Stint, LapNumber,
            FuelLoad (estimated linear decrease), AvgThrottle, AvgBrake, IsSC
        """
        df = laps.copy()

        # LapTime → seconds
        if "LapTime_sec" not in df.columns:
            if "LapTime" in df.columns:
                if np.issubdtype(df["LapTime"].dtype, np.timedelta64):
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

        # Compound handling — fill NaN before encoding
        if "Compound" not in df.columns:
            df["Compound"] = "UNKNOWN"
        df["Compound"] = df["Compound"].fillna("UNKNOWN")

        # LabelEncoder for backward compatibility (Rob's code)
        le = LabelEncoder()
        df["Compound_encoded"] = le.fit_transform(df["Compound"].astype(str).str.upper())

        # Also create Compound_enc via static map for merged model consistency
        df["Compound_enc"] = (
            df["Compound"].str.upper().map(ModelEngine.COMPOUND_MAP).fillna(1).astype(int)
        )

        # TyreLife default
        if "TyreLife" not in df.columns:
            df["TyreLife"] = 1
        df["TyreLife"] = df["TyreLife"].fillna(1)

        # Stint default
        if "Stint" not in df.columns:
            df["Stint"] = 1
        df["Stint"] = df["Stint"].fillna(1)

        # LapNumber default
        if "LapNumber" not in df.columns:
            df["LapNumber"] = range(1, len(df) + 1)

        # Fuel load estimate (linear decrease, starting from ~110 kg)
        max_lap = df["LapNumber"].max()
        if max_lap > 0:
            df["FuelLoad"] = 110.0 * (1.0 - df["LapNumber"] / max_lap)
        else:
            df["FuelLoad"] = 110.0

        # Fill optional telemetry columns
        for col in ("AvgThrottle", "AvgBrake", "IsSC"):
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0)

        return df.reset_index(drop=True), le

    # -----------------------------------------------------------------------
    # XGBoost training
    # -----------------------------------------------------------------------

    FEATURE_COLS: List[str] = [
        "LapNumber",
        "TyreLife",
        "Compound_encoded",
        "Stint",
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
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)
        return model

    # -----------------------------------------------------------------------
    # Autoregressive prediction (Series-based – used by run_e2e.py)
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
        """Autoregressively predict future lap times from a Series row.

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
        compound_enc = ModelEngine.COMPOUND_MAP.get(compound.upper(), 1)

        feature_cols = [c for c in ModelEngine.FEATURE_COLS if c in last_known.index]
        predictions: List[Dict[str, Any]] = []
        prev_time = float(last_known.get("LapTime_sec", last_known.get("Prev_LapTime_sec", 90)))

        start_lap = int(last_known.get("LapNumber", 0)) + 1

        for i in range(laps_ahead):
            lap_num = start_lap + i
            tyre_life = tyre_life_start + i
            # Fuel load relative to a typical race length (~60 laps)
            fuel_load = max(0.0, 110.0 * (1.0 - lap_num / 70.0))

            row: Dict[str, float] = {
                "LapNumber": lap_num,
                "TyreLife": tyre_life,
                "Compound_encoded": compound_enc,
                "Stint": stint,
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
    # Autoregressive prediction (scalar-based – used by app.py, Rob's style)
    # -----------------------------------------------------------------------

    @staticmethod
    def predict_future_pace_from_state(
        model: xgb.XGBRegressor,
        current_lap: int,
        current_tyre_life: int,
        current_compound_enc: int,
        current_stint: int,
        current_laptime: float,
        avg_throttle: float,
        avg_brake: float,
        num_laps: int = 5,
    ) -> pd.DataFrame:
        """Autoregressively predict future lap times from scalar state values.

        This is the interface used by app.py and compatible with Rob's
        ``predict_future_pace`` function signature.

        Parameters
        ----------
        model : XGBRegressor
            Fitted pace model.
        current_lap, current_tyre_life, current_compound_enc, current_stint :
            Current race state scalars.
        current_laptime : float
            Last known lap time in seconds.
        avg_throttle, avg_brake : float
            Average throttle/brake percentages.
        num_laps : int
            Number of future laps to predict.

        Returns
        -------
        DataFrame
            Columns: LapNumber, TyreLife, Compound_encoded, Stint,
            Predicted_LapTime_sec.
        """
        future_data: List[Dict[str, Any]] = []
        prev_lt = current_laptime

        feature_cols = ModelEngine.FEATURE_COLS

        for i in range(1, num_laps + 1):
            row = {
                "LapNumber": current_lap + i,
                "TyreLife": current_tyre_life + i,
                "Compound_encoded": current_compound_enc,
                "Stint": current_stint,
                "Prev_LapTime_sec": prev_lt,
                "AvgThrottle": avg_throttle,
                "AvgBrake": avg_brake,
                "IsSC": 0,
            }

            features_df = pd.DataFrame([row])
            # Use only the columns the model was trained on
            available_cols = [c for c in feature_cols if c in features_df.columns]
            pred = float(model.predict(features_df[available_cols])[0])

            future_data.append({
                "LapNumber": current_lap + i,
                "TyreLife": current_tyre_life + i,
                "Compound_encoded": current_compound_enc,
                "Stint": current_stint,
                "Predicted_LapTime_sec": pred,
            })

            # Autoregression: prediction becomes previous lap time for next lap
            prev_lt = pred

        return pd.DataFrame(future_data)

    # -----------------------------------------------------------------------
    # Win probability (rich output for the Streamlit dashboard)
    # -----------------------------------------------------------------------

    @staticmethod
    def get_real_win_probability(
        lap_metrics_df: pd.DataFrame,
        pace_models: Dict[str, xgb.XGBRegressor],
        remaining_laps: int = 10,
    ) -> List[Dict[str, Any]]:
        """Estimate win probability and produce rich per-driver stats.

        This combines position-based scoring with optional pace model
        predictions and strategic insights (pit window, fuel, tyre
        degradation trend).

        Parameters
        ----------
        lap_metrics_df : DataFrame
            Latest lap metrics with at least ``Driver`` and ``Position``
            columns.  Optionally: ``LapTime`` / ``LapTime_sec``,
            ``Compound``, ``TyreLife``, ``SpeedST``, ``TrackStatus``,
            ``Stint``, ``LapNumber``, ``TotalLaps``.
        pace_models : dict
            ``{driver_abbreviation: fitted_XGBRegressor}``.
        remaining_laps : int
            Number of laps remaining in the race.

        Returns
        -------
        list[dict]
            Each dict contains: Posizione, Pilota, Probabilità, Andamento,
            Finestra Pit, Carburante, Pit Stop, Accelerazione, Tempo, Gomma,
            Età Gomma, Vel. Max, Traffico (Stato).
        """
        if lap_metrics_df.empty or "Driver" not in lap_metrics_df.columns:
            return []

        df = lap_metrics_df.copy()

        # Ensure Position column
        if "Position" not in df.columns:
            df["Position"] = range(1, len(df) + 1)

        df = df.sort_values("Position")

        # --- Step 1: compute probability scores ---
        total_score = 0.0
        scores: List[float] = []

        for _, row in df.iterrows():
            pos = row.get("Position", 20)
            try:
                pos = float(pos)
            except (ValueError, TypeError):
                pos = 20.0
            if pd.isna(pos):
                pos = 20.0
            # Exponential position-based scoring
            score = 1.0 / (pos ** 1.5)

            # Bonus from pace model if available
            driver = row["Driver"]
            if driver in pace_models:
                try:
                    mdl = pace_models[driver]
                    pred_row = row.copy()
                    if "LapTime_sec" not in pred_row or pd.isna(pred_row.get("LapTime_sec")):
                        if "LapTime" in pred_row:
                            lt = pred_row["LapTime"]
                            if hasattr(lt, "total_seconds"):
                                pred_row["LapTime_sec"] = lt.total_seconds()
                    pred = ModelEngine.predict_future_pace(
                        mdl, pred_row, laps_ahead=min(remaining_laps, 5),
                    )
                    if not pred.empty:
                        avg_predicted = pred["Predicted_LapTime_sec"].mean()
                        pace_bonus = max(0.0, 100.0 - avg_predicted) * 0.01
                        score += pace_bonus
                except Exception:
                    pass

            scores.append(max(score, 0.001))
            total_score += scores[-1]

        # Normalise
        probabilities = [s / total_score for s in scores] if total_score > 0 else [1.0 / len(scores)] * len(scores)

        # --- Step 2: build rich result rows ---
        results: List[Dict[str, Any]] = []

        for idx, (_, row) in enumerate(df.iterrows()):
            driver = row["Driver"]
            prob = probabilities[idx]

            pos = row.get("Position", 20)
            try:
                pos_int = int(float(pos)) if not pd.isna(pos) else 20
            except (ValueError, TypeError):
                pos_int = 20

            # Lap time
            lap_time = row.get("LapTime_sec", None)
            if lap_time is None or (isinstance(lap_time, float) and pd.isna(lap_time)):
                # Try converting from timedelta
                lt_raw = row.get("LapTime", None)
                if lt_raw is not None and hasattr(lt_raw, "total_seconds"):
                    lap_time = lt_raw.total_seconds()
                else:
                    try:
                        lap_time = float(lt_raw) if lt_raw is not None else 0
                    except (ValueError, TypeError):
                        lap_time = 0

            if pd.isna(lap_time) or lap_time == 0:
                lap_time_str = "N/A"
            else:
                m = int(lap_time // 60)
                s = lap_time % 60
                lap_time_str = f"{m}:{s:06.3f}"

            # Compound / tyre
            compound = row.get("Compound", "UNKNOWN")
            if pd.isna(compound):
                compound = "UNKNOWN"
            compound = str(compound)

            tyre_life = row.get("TyreLife", 0)
            try:
                tyre_life = int(tyre_life) if not pd.isna(tyre_life) else 0
            except (ValueError, TypeError):
                tyre_life = 0

            # Speed
            speed = row.get("SpeedST", None)
            if speed is not None and not (isinstance(speed, float) and pd.isna(speed)):
                speed_str = f"{speed} km/h"
            else:
                speed_str = "N/A"

            # Track status
            traffic = row.get("TrackStatus", "1")
            if pd.isna(traffic):
                traffic = "1"
            traffic_str = "Libero" if str(traffic) == "1" else "Traffico/Bandiera"

            # Stint / pit stops
            stint = row.get("Stint", 1)
            try:
                stint_int = int(stint) if not pd.isna(stint) else 1
            except (ValueError, TypeError):
                stint_int = 1
            pit_stops = max(0, stint_int - 1)

            # Lap number & total laps for fuel estimate
            lap_num = row.get("LapNumber", 1)
            try:
                lap_num = int(lap_num) if not pd.isna(lap_num) else 1
            except (ValueError, TypeError):
                lap_num = 1

            total_laps = row.get("TotalLaps", 70)
            try:
                total_laps = int(total_laps) if not pd.isna(total_laps) else 70
            except (ValueError, TypeError):
                total_laps = 70

            # Fuel load estimation (starts at ~110 kg, ends at ~2 kg)
            fuel_kg = 110 - ((110 - 2) * (lap_num / total_laps)) if total_laps > 0 else 50.0

            # Pit window estimation
            undercut_range = "Chiusa"
            if compound == "SOFT" and tyre_life > 12:
                undercut_range = "Aperta (Undercut)"
            elif compound == "MEDIUM" and tyre_life > 20:
                undercut_range = "Aperta (Undercut)"
            elif compound == "HARD" and tyre_life > 35:
                undercut_range = "Aperta (Undercut)"
            elif tyre_life < 5 and pit_stops > 0:
                undercut_range = "Tentativo Overcut"

            # Acceleration estimation
            accel = "Alta" if compound == "SOFT" else ("Media" if compound == "MEDIUM" else "Bassa")

            # Pace trend
            trend = "Costante"
            if tyre_life > 20:
                trend = "In calo (Degrado)"
            elif tyre_life < 5:
                trend = "In miglioramento"
            if pos_int == 1 and tyre_life < 15:
                trend = "Dominante"

            results.append({
                "Posizione": pos_int,
                "Pilota": driver,
                "Probabilità": prob,
                "Andamento": trend,
                "Finestra Pit": undercut_range,
                "Carburante": f"{fuel_kg:.1f} kg",
                "Pit Stop": pit_stops,
                "Accelerazione": accel,
                "Tempo": lap_time_str,
                "Gomma": compound,
                "Età Gomma": tyre_life,
                "Vel. Max": speed_str,
                "Traffico (Stato)": traffic_str,
            })

        return results
