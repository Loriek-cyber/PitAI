#!/usr/bin/env python
"""
run_e2e.py – End-to-end integration test / demo runner for the merged F1 project.

Exercises both data backends (OpenF1 + FastF1) and the ML modelling pipeline.
Colored terminal output via ``termcolor``.

Usage:
    python run_e2e.py --year 2024 --race "Monza" --driver VER
    python run_e2e.py --year 2024 --race "Monza" --driver VER --session-key 9574
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import Optional

import pandas as pd
from termcolor import colored

from data_core import DataCore
from models import ModelEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    """Print a section header."""
    sep = "=" * 60
    print(colored(f"\n{sep}", "cyan"))
    print(colored(f"  {title}", "cyan", attrs=["bold"]))
    print(colored(sep, "cyan"))


def ok(msg: str) -> None:
    print(colored(f"  ✓ {msg}", "green"))


def warn(msg: str) -> None:
    print(colored(f"  ⚠ {msg}", "yellow"))


def fail(msg: str) -> None:
    print(colored(f"  ✗ {msg}", "red"))


# ---------------------------------------------------------------------------
# Test steps
# ---------------------------------------------------------------------------

def test_schedule(dc: DataCore, year: int) -> None:
    header(f"Schedule – {year}")
    try:
        sched = dc.get_schedule(year)
        ok(f"FastF1 schedule: {len(sched)} events")
    except Exception as exc:
        fail(f"FastF1 schedule failed: {exc}")

    try:
        races = dc.get_available_races(year)
        ok(f"OpenF1 races: {len(races)} sessions")
    except Exception as exc:
        fail(f"OpenF1 races failed: {exc}")


def test_drivers(dc: DataCore, year: int, race: str, session_key: Optional[int]) -> None:
    header(f"Drivers – {year} {race}")
    drivers = dc.get_drivers(year, race, session_key=session_key)
    if drivers:
        ok(f"Drivers ({len(drivers)}): {', '.join(drivers[:10])}{'...' if len(drivers) > 10 else ''}")
    else:
        warn("No drivers found")


def test_total_laps(dc: DataCore, year: int, race: str, session_key: Optional[int]) -> None:
    header(f"Total laps – {year} {race}")
    total = dc.get_total_laps(year, race, session_key=session_key)
    if total > 0:
        ok(f"Total laps: {total}")
    else:
        warn("Could not determine total laps")


def test_laps(dc: DataCore, year: int, race: str, driver: str, session_key: Optional[int]) -> pd.DataFrame:
    header(f"Laps – {driver} @ {year} {race}")
    df = dc.get_laps(year, race, driver, session_key=session_key)
    if not df.empty:
        ok(f"Got {len(df)} laps")
        # Show a few columns if available
        cols_show = [c for c in ["LapNumber", "LapTime", "Compound", "TyreLife"] if c in df.columns]
        if cols_show:
            print(df[cols_show].head(5).to_string(index=False))
    else:
        warn("No laps returned")
    return df


def test_openf1_laps(dc: DataCore, session_key: int, driver: str) -> pd.DataFrame:
    header(f"OpenF1 laps (with stints/car_data/SC) – {driver}")
    df = dc.get_race_laps(session_key, driver)
    if not df.empty:
        ok(f"Got {len(df)} laps with columns: {list(df.columns)}")
        print(df.head(5).to_string(index=False))
    else:
        warn("No OpenF1 laps returned")
    return df


def test_telemetry(dc: DataCore, year: int, race: str, driver: str, session_key: Optional[int]) -> None:
    header(f"Telemetry – {driver} @ {year} {race}")
    fastest, tel = dc.get_telemetry(year, race, driver, session_key=session_key)
    if not tel.empty:
        ok(f"Telemetry points: {len(tel)}")
        if fastest:
            ok(f"Fastest lap duration: {fastest.get('lap_duration', 'N/A')}s")
    else:
        warn("No telemetry returned")


def test_head_to_head(dc: DataCore, year: int, race: str, d1: str, d2: str) -> None:
    header(f"Head-to-Head – {d1} vs {d2}")
    delta = dc.fetch_driver_head_to_head(d1, d2, year, race)
    if not delta.empty:
        ok(f"Delta rows: {len(delta)}")
        print(delta.head(5).to_string(index=False))
    else:
        warn("Head-to-head data unavailable")


def test_model_pipeline(laps_df: pd.DataFrame, driver: str) -> None:
    header(f"ML Pipeline – {driver}")
    me = ModelEngine()

    if laps_df.empty:
        warn("No laps to train on – skipping ML pipeline")
        return

    # Feature engineering
    features_df = me.prepare_pace_features(laps_df)
    ok(f"Features prepared: {len(features_df)} rows × {len(features_df.columns)} cols")

    if len(features_df) < 5:
        warn("Too few data points for training – skipping")
        return

    # Train XGBoost
    model = me.train_pace_model(features_df)
    ok("XGBoost model trained")

    # Predict future pace
    last = features_df.iloc[-1]
    preds = me.predict_future_pace(model, last, laps_ahead=5)
    ok(f"Predicted {len(preds)} future laps")
    if not preds.empty:
        print(preds[["LapNumber", "Predicted_LapTime_sec"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="F1 Merged E2E runner")
    parser.add_argument("--year", type=int, default=2024, help="Season year")
    parser.add_argument("--race", type=str, default="Monza", help="GP name")
    parser.add_argument("--driver", type=str, default="VER", help="Driver abbreviation")
    parser.add_argument("--driver2", type=str, default="HAM", help="Second driver for head-to-head")
    parser.add_argument("--session-key", type=int, default=None, help="OpenF1 session key (optional)")
    args = parser.parse_args()

    print(colored("\n🏎  F1 Merged Project – End-to-End Test Run", "white", attrs=["bold"]))
    print(colored(f"   Year={args.year}  Race={args.race}  Driver={args.driver}\n", "white"))

    dc = DataCore()
    t0 = time.time()

    try:
        # Schedule
        test_schedule(dc, args.year)

        # Session key resolution
        sk = args.session_key
        if sk is None:
            sk = dc.load_session(args.year, args.race)
            if sk:
                ok(f"Resolved session_key = {sk}")
            else:
                warn("Could not resolve session_key from OpenF1")

        # Drivers
        test_drivers(dc, args.year, args.race, sk)

        # Total laps
        test_total_laps(dc, args.year, args.race, sk)

        # Laps (unified)
        laps = test_laps(dc, args.year, args.race, args.driver, sk)

        # OpenF1 laps (full implementation with stints/SC)
        openf1_laps = pd.DataFrame()
        if sk is not None:
            openf1_laps = test_openf1_laps(dc, sk, args.driver)

        # Telemetry
        test_telemetry(dc, args.year, args.race, args.driver, sk)

        # Head-to-head
        test_head_to_head(dc, args.year, args.race, args.driver, args.driver2)

        # ML pipeline (prefer OpenF1 laps for richer features)
        ml_laps = openf1_laps if not openf1_laps.empty else laps
        test_model_pipeline(ml_laps, args.driver)

    except KeyboardInterrupt:
        print(colored("\n\nInterrupted by user.", "yellow"))
        sys.exit(1)
    except Exception:
        fail("Unexpected error:")
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    print(colored(f"\n{'=' * 60}", "cyan"))
    print(colored(f"  Done in {elapsed:.1f}s  |  Network hit: {dc.network_hit}", "cyan"))
    print(colored(f"{'=' * 60}\n", "cyan"))


if __name__ == "__main__":
    main()
