"""
data_core.py – Unified data layer for the merged F1 project.

Merges:
  - Rob's OpenF1 API-based data_handler (pickle cache, retry, stints, car_data, track_status)
  - Mine's FastF1-based data_pipeline (schedule, drivers, laps, telemetry, head-to-head)

Strategy: try FastF1 first, fall back to OpenF1 on failure.
"""

from __future__ import annotations

import os
import pickle
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import fastf1 as ff1

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grand-Prix name mapping (OpenF1 uses country names for some GPs)
# ---------------------------------------------------------------------------
GP_MAPPING: Dict[str, str] = {
    "Silverstone": "Great Britain",
    "Spa": "Belgium",
    "Monza": "Italy",
    "Monaco": "Monaco",
    "Imola": "Emilia Romagna",
    "Baku": "Azerbaijan",
    "Jeddah": "Saudi Arabia",
    "Lusail": "Qatar",
    "Yas Marina": "Abu Dhabi",
    "Suzuka": "Japan",
    "Shanghai": "China",
    "Melbourne": "Australia",
    "Interlagos": "Brazil",
    "Sao Paulo": "Brazil",
    "Zandvoort": "Netherlands",
    "Hungaroring": "Hungary",
    "Barcelona": "Spain",
    "Montreal": "Canada",
    "Austin": "United States",
    "COTA": "United States",
    "Miami": "Miami",
    "Las Vegas": "Las Vegas",
    "Singapore": "Singapore",
    "Mexico City": "Mexico",
}


class MockResponse:
    """Fallback response object for 404 / failed HTTP calls."""

    def __init__(self) -> None:
        self.status_code: int = 404

    def json(self) -> list:  # noqa: D401
        return []


class DataCore:
    """Unified data layer combining OpenF1 (REST + pickle cache) and FastF1."""

    # OpenF1 base URL
    OPENF1_BASE = "https://api.openf1.org/v1"

    def __init__(
        self,
        cache_dir: str = "./cache_data",
        ff1_cache_dir: str = "./f1_local_cache",
    ) -> None:
        # Pickle cache directory (OpenF1 data)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # FastF1 cache directory
        self.ff1_cache_dir = ff1_cache_dir
        os.makedirs(self.ff1_cache_dir, exist_ok=True)
        ff1.Cache.enable_cache(self.ff1_cache_dir)

        # Flag: True when a network call was actually made (not served from cache)
        self.network_hit: bool = False

    # =======================================================================
    #  OpenF1 low-level helpers
    # =======================================================================

    def _requests_get_with_retry(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> requests.Response:
        """HTTP GET with exponential back-off for 429 / 5xx errors."""
        self.network_hit = True
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = 2 ** attempt
                    logger.warning(
                        "HTTP %s from %s – retrying in %ss (attempt %d/%d)",
                        resp.status_code,
                        url,
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait)
                    continue
                # Other non-200: return a mock
                logger.warning("HTTP %s from %s – returning empty", resp.status_code, url)
                return MockResponse()  # type: ignore[return-value]
            except requests.RequestException as exc:
                logger.warning("Request error: %s – retrying (%d/%d)", exc, attempt + 1, max_retries)
                time.sleep(2 ** attempt)
        return MockResponse()  # type: ignore[return-value]

    def _cached_api_call(
        self,
        cache_name: str,
        fetch_func: Callable[[], Any],
    ) -> Any:
        """Pickle-based disk caching wrapper."""
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                pass

        data = fetch_func()
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(data, fh)
        except Exception:
            pass
        return data

    # =======================================================================
    #  OpenF1 session / driver helpers
    # =======================================================================

    def get_available_races(self, year: int) -> List[Dict[str, Any]]:
        """Return list of race sessions from OpenF1 for *year*."""
        def _fetch() -> List[Dict[str, Any]]:
            resp = self._requests_get_with_retry(
                f"{self.OPENF1_BASE}/sessions",
                params={"year": year, "session_type": "Race"},
            )
            data = resp.json()
            return data if isinstance(data, list) else []

        return self._cached_api_call(f"races_{year}", _fetch)

    def get_session_drivers(self, session_key: int) -> List[str]:
        """Return driver acronyms participating in *session_key*."""
        def _fetch() -> List[str]:
            resp = self._requests_get_with_retry(
                f"{self.OPENF1_BASE}/drivers",
                params={"session_key": session_key},
            )
            data = resp.json()
            if isinstance(data, dict):
                data = []
            seen: set[str] = set()
            drivers: List[str] = []
            for d in data:
                acr = d.get("name_acronym", "")
                if acr and acr not in seen:
                    seen.add(acr)
                    drivers.append(acr)
            return drivers

        return self._cached_api_call(f"drivers_{session_key}", _fetch)

    def get_driver_number(self, session_key: int, driver_acronym: str) -> Optional[int]:
        """Map a driver acronym to their car number for *session_key*."""
        resp = self._requests_get_with_retry(
            f"{self.OPENF1_BASE}/drivers",
            params={"session_key": session_key, "name_acronym": driver_acronym},
        )
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("driver_number")
        return None

    def get_race_total_laps(self, session_key: int) -> int:
        """Return the maximum lap number recorded for *session_key*."""
        resp = self._requests_get_with_retry(
            f"{self.OPENF1_BASE}/laps",
            params={"session_key": session_key},
        )
        data = resp.json()
        if isinstance(data, dict):
            data = []
        if not data:
            return 0
        return max(l.get("lap_number", 0) for l in data)

    def get_qualy_fastest_lap(
        self,
        qualy_session_key: int,
        driver_acronym: str,
    ) -> Optional[float]:
        """Return the fastest lap duration (seconds) in qualifying."""
        driver_no = self.get_driver_number(qualy_session_key, driver_acronym)
        if driver_no is None:
            return None
        resp = self._requests_get_with_retry(
            f"{self.OPENF1_BASE}/laps",
            params={"session_key": qualy_session_key, "driver_number": driver_no},
        )
        data = resp.json()
        if isinstance(data, dict):
            data = []
        durations = [l["lap_duration"] for l in data if l.get("lap_duration")]
        return min(durations) if durations else None

    def load_session(
        self,
        year: int,
        grand_prix: str,
        session_type: str = "Race",
    ) -> Optional[int]:
        """Resolve an OpenF1 session_key from year + GP name + session type.

        Uses *GP_MAPPING* for common aliases (e.g. "Silverstone" → "Great Britain").
        """
        search_name = GP_MAPPING.get(grand_prix, grand_prix)
        resp = self._requests_get_with_retry(
            f"{self.OPENF1_BASE}/sessions",
            params={"year": year, "session_type": session_type},
        )
        data = resp.json()
        if isinstance(data, dict):
            data = []

        for s in data:
            meeting_name = s.get("meeting_name", "") or ""
            country = s.get("country_name", "") or ""
            location = s.get("location", "") or ""
            if (
                search_name.lower() in meeting_name.lower()
                or search_name.lower() in country.lower()
                or search_name.lower() in location.lower()
                or grand_prix.lower() in meeting_name.lower()
                or grand_prix.lower() in location.lower()
            ):
                return s.get("session_key")
        return None

    # =======================================================================
    #  OpenF1 race laps (FULL implementation with stints, car_data, track_status)
    # =======================================================================

    def get_race_laps(
        self,
        session_key: int,
        driver: str,
        is_live: bool = False,
    ) -> pd.DataFrame:
        """Fetch race laps from OpenF1, enriched with stint info, car telemetry
        averages (AvgThrottle, AvgBrake) and safety-car flag (IsSC).

        Results are pickle-cached unless *is_live* is True.
        """
        cache_path = os.path.join(self.cache_dir, f"laps_{session_key}_{driver}.pkl")
        if not is_live and os.path.exists(cache_path):
            try:
                return pd.read_pickle(cache_path)
            except Exception:
                pass

        driver_no = self.get_driver_number(session_key, driver)

        # --- Laps ---
        res = self._requests_get_with_retry(
            f"{self.OPENF1_BASE}/laps",
            params={"session_key": session_key, "driver_number": driver_no},
        )
        laps_data = res.json()
        if isinstance(laps_data, dict):
            laps_data = []
        df_laps = pd.DataFrame(laps_data)

        # --- Stints ---
        res_stints = self._requests_get_with_retry(
            f"{self.OPENF1_BASE}/stints",
            params={"session_key": session_key, "driver_number": driver_no},
        )
        stints_data = res_stints.json()
        if isinstance(stints_data, dict):
            stints_data = []

        # --- Car data (throttle / brake) ---
        res_car = self._requests_get_with_retry(
            f"{self.OPENF1_BASE}/car_data",
            params={"session_key": session_key, "driver_number": driver_no},
        )
        car_data = res_car.json()
        if isinstance(car_data, dict):
            car_data = []
        if car_data:
            df_car = pd.DataFrame(car_data)
            df_car["date"] = pd.to_datetime(df_car["date"], format="ISO8601")
        else:
            df_car = pd.DataFrame(columns=["date", "throttle", "brake"])

        # --- Track status (safety car detection) ---
        res_ts = self._requests_get_with_retry(
            f"{self.OPENF1_BASE}/track_status",
            params={"session_key": session_key},
        )
        ts_data = res_ts.json()
        if isinstance(ts_data, dict):
            ts_data = []
        if ts_data:
            df_ts = pd.DataFrame(ts_data)
            df_ts["date"] = pd.to_datetime(df_ts["date"], format="ISO8601")
            df_ts = df_ts.sort_values("date")
        else:
            df_ts = pd.DataFrame(columns=["date", "status"])

        # --- Assemble output rows ---
        out: List[Dict[str, Any]] = []
        tyre_age = 1
        current_stint = 1
        current_compound = "SOFT"
        stint_idx = 0

        for _, row in df_laps.iterrows():
            lap_num = row.get("lap_number", 0)

            # Stint tracking
            if stint_idx < len(stints_data):
                st = stints_data[stint_idx]
                lap_start = st.get("lap_start")
                lap_start = lap_start if lap_start is not None else 0
                lap_end = st.get("lap_end")
                lap_end = lap_end if lap_end is not None else 999
                tyre_start = st.get("tyre_age_at_start")
                tyre_start = tyre_start if tyre_start is not None else 0

                if lap_start <= lap_num <= lap_end:
                    current_stint = st.get("stint_number", 1)
                    current_compound = st.get("compound", "UNKNOWN")
                    tyre_age = tyre_start + (lap_num - lap_start)
                elif lap_num > lap_end:
                    stint_idx += 1
                    if stint_idx < len(stints_data):
                        st = stints_data[stint_idx]
                        lap_start = st.get("lap_start")
                        lap_start = lap_start if lap_start is not None else 0
                        tyre_start = st.get("tyre_age_at_start")
                        tyre_start = tyre_start if tyre_start is not None else 0
                        current_stint = st.get("stint_number", 1)
                        current_compound = st.get("compound", "UNKNOWN")
                        tyre_age = tyre_start + (lap_num - lap_start)

            ld = row.get("lap_duration")
            ds = row.get("date_start")

            avg_thr = 0.0
            avg_brk = 0.0
            is_sc = 0

            if pd.notna(ld) and ds:
                ds_dt = pd.to_datetime(ds, format="ISO8601")
                de_dt = ds_dt + pd.to_timedelta(ld, unit="s")

                if not df_car.empty:
                    lap_car = df_car[
                        (df_car["date"] >= ds_dt) & (df_car["date"] <= de_dt)
                    ]
                    if not lap_car.empty:
                        avg_thr = lap_car["throttle"].mean()
                        avg_brk = lap_car["brake"].mean()

                if not df_ts.empty:
                    prev_statuses = df_ts[df_ts["date"] <= de_dt]
                    if not prev_statuses.empty:
                        last_status = prev_statuses.iloc[-1]["status"]
                        if str(last_status) in ("4", "5"):
                            is_sc = 1

            out.append(
                {
                    "LapNumber": lap_num,
                    "LapTime": pd.to_timedelta(ld, unit="s") if ld else pd.NaT,
                    "Sector1Time": (
                        pd.to_timedelta(row.get("duration_sector_1"), unit="s")
                        if row.get("duration_sector_1")
                        else pd.NaT
                    ),
                    "Sector2Time": (
                        pd.to_timedelta(row.get("duration_sector_2"), unit="s")
                        if row.get("duration_sector_2")
                        else pd.NaT
                    ),
                    "Sector3Time": (
                        pd.to_timedelta(row.get("duration_sector_3"), unit="s")
                        if row.get("duration_sector_3")
                        else pd.NaT
                    ),
                    "TyreLife": tyre_age,
                    "Compound": current_compound,
                    "Stint": current_stint,
                    "IsAccurate": pd.notna(ld),
                    "date_start": row.get("date_start"),
                    "AvgThrottle": avg_thr,
                    "AvgBrake": avg_brk,
                    "IsSC": is_sc,
                }
            )

        df_out = pd.DataFrame(out)
        if not is_live and not df_out.empty:
            df_out.to_pickle(cache_path)
        return df_out

    # =======================================================================
    #  OpenF1 telemetry for fastest lap
    # =======================================================================

    def get_fastest_lap_telemetry(
        self,
        session_key: int,
        driver: str,
        is_live: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
        """Return ``(fastest_lap_dict, telemetry_df)`` for the fastest lap.

        Telemetry includes computed *Distance* from speed integration.
        """
        cache_path = os.path.join(self.cache_dir, f"tel_{session_key}_{driver}.pkl")
        if not is_live and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                pass

        driver_no = self.get_driver_number(session_key, driver)
        res = self._requests_get_with_retry(
            f"{self.OPENF1_BASE}/laps",
            params={"session_key": session_key, "driver_number": driver_no},
        )
        data = res.json()
        if isinstance(data, dict):
            data = []
        laps_data = [l for l in data if l.get("lap_duration")]

        if not laps_data:
            return None, pd.DataFrame()

        fastest = min(laps_data, key=lambda x: x["lap_duration"])
        start_ts = pd.to_datetime(fastest["date_start"], format="ISO8601")
        end_ts = start_ts + pd.to_timedelta(fastest["lap_duration"], unit="s")

        start_ts_str = start_ts.isoformat().replace("+", "%2B")
        end_ts_str = end_ts.isoformat().replace("+", "%2B")
        url = (
            f"{self.OPENF1_BASE}/car_data?"
            f"session_key={session_key}&driver_number={driver_no}"
            f"&date>={start_ts_str}&date<={end_ts_str}"
        )

        res_car = self._requests_get_with_retry(url)
        car_data = res_car.json()
        if isinstance(car_data, dict):
            car_data = []
        df_car = pd.DataFrame(car_data)

        if df_car.empty:
            return fastest, pd.DataFrame()

        df_car["date"] = pd.to_datetime(df_car["date"], format="ISO8601")
        df_car = df_car.sort_values("date")
        df_car["Time_diff"] = df_car["date"].diff().dt.total_seconds().fillna(0)
        df_car["Speed_ms"] = df_car["speed"] / 3.6
        df_car["Distance"] = (df_car["Speed_ms"] * df_car["Time_diff"]).cumsum()
        df_car = df_car.rename(columns={"speed": "Speed"})

        if not is_live and not df_car.empty:
            with open(cache_path, "wb") as fh:
                pickle.dump((fastest, df_car), fh)
        return fastest, df_car

    # =======================================================================
    #  FastF1-based methods
    # =======================================================================

    def get_schedule(self, year: int) -> pd.DataFrame:
        """Return the event schedule for *year* via FastF1.

        Testing events are excluded.
        """
        schedule = ff1.get_event_schedule(year)
        if "EventFormat" in schedule.columns:
            schedule = schedule[schedule["EventFormat"] != "testing"]
        return schedule

    def get_drivers_ff1(self, year: int, race: str | int) -> List[str]:
        """Return driver abbreviations for a race via FastF1 session results."""
        try:
            session = ff1.get_session(year, race, "R")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            results = session.results
            if results is not None and not results.empty:
                return results["Abbreviation"].tolist()
        except Exception as exc:
            logger.warning("FastF1 get_drivers_ff1 failed: %s", exc)
        return []

    def get_total_laps_ff1(self, year: int, race: str | int) -> int:
        """Return the total number of laps for a race via FastF1."""
        try:
            session = ff1.get_session(year, race, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            return int(session.laps["LapNumber"].max())
        except Exception as exc:
            logger.warning("FastF1 get_total_laps failed: %s", exc)
        return 0

    def get_lap_metrics(
        self,
        year: int,
        race: str | int,
        lap_number: int,
    ) -> pd.DataFrame:
        """Return cross-driver metrics for a specific lap via FastF1.

        Columns include: Driver, Position, SpeedST, TrackStatus, Stint,
        LapTime, Compound, TyreLife, LapTime_sec, TotalLaps, LapNumber.
        """
        try:
            session = ff1.get_session(year, race, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            lap_df = session.laps[session.laps["LapNumber"] == lap_number].copy()
            if lap_df.empty:
                return pd.DataFrame()

            cols_keep = []
            for col in ["Driver", "Position", "SpeedST", "TrackStatus", "Stint", "LapTime", "Compound", "TyreLife"]:
                if col in lap_df.columns:
                    cols_keep.append(col)

            result = lap_df[cols_keep].reset_index(drop=True)

            # Enrich with derived columns needed by ModelEngine.get_real_win_probability
            if "LapTime" in result.columns:
                result["LapTime_sec"] = result["LapTime"].dt.total_seconds()

            # Total laps for fuel estimation
            total_laps = int(session.total_laps) if hasattr(session, "total_laps") and session.total_laps else 70
            result["TotalLaps"] = total_laps
            result["LapNumber"] = lap_number

            return result
        except Exception as exc:
            logger.warning("FastF1 get_lap_metrics failed: %s", exc)
        return pd.DataFrame()

    def get_race_laps_ff1(
        self,
        year: int,
        race: str | int,
        driver: str,
    ) -> pd.DataFrame:
        """Return race laps for *driver* via FastF1."""
        try:
            session = ff1.get_session(year, race, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = session.laps.pick_drivers(driver)
            return laps.reset_index(drop=True)
        except Exception as exc:
            logger.warning("FastF1 get_race_laps_ff1 failed: %s", exc)
        return pd.DataFrame()

    def get_telemetry_ff1(
        self,
        year: int,
        race: str | int,
        driver: str,
    ) -> pd.DataFrame:
        """Return telemetry for the fastest lap via FastF1."""
        try:
            session = ff1.get_session(year, race, "R")
            session.load(laps=True, telemetry=True, weather=False, messages=False)
            driver_laps = session.laps.pick_drivers(driver)
            fastest = driver_laps.pick_fastest()
            telemetry = fastest.get_telemetry()
            return telemetry.reset_index(drop=True)
        except Exception as exc:
            logger.warning("FastF1 get_telemetry failed: %s", exc)
        return pd.DataFrame()

    def fetch_driver_head_to_head(
        self,
        driver1: str,
        driver2: str,
        year: int,
        race: str | int,
    ) -> pd.DataFrame:
        """Compute lap-by-lap delta between two drivers via FastF1.

        Returns a DataFrame with columns: LapNumber, Delta (driver1 - driver2 in seconds).
        """
        try:
            session = ff1.get_session(year, race, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)

            laps1 = session.laps.pick_drivers(driver1)[["LapNumber", "LapTime"]].copy()
            laps2 = session.laps.pick_drivers(driver2)[["LapNumber", "LapTime"]].copy()

            laps1 = laps1.rename(columns={"LapTime": "LapTime_D1"})
            laps2 = laps2.rename(columns={"LapTime": "LapTime_D2"})

            merged = pd.merge(laps1, laps2, on="LapNumber", how="inner")
            merged["Delta"] = (
                merged["LapTime_D1"].dt.total_seconds()
                - merged["LapTime_D2"].dt.total_seconds()
            )
            result = merged[["LapNumber", "Delta"]].reset_index(drop=True)
            # Add 'Lap' alias for backward compatibility with app.py
            result["Lap"] = result["LapNumber"]
            return result
        except Exception as exc:
            logger.warning("FastF1 head-to-head failed: %s", exc)
        return pd.DataFrame()

    # =======================================================================
    #  Unified convenience methods (FastF1 first, OpenF1 fallback)
    # =======================================================================

    def get_drivers(
        self,
        year: int,
        race: str | int,
        session_key: Optional[int] = None,
    ) -> List[str]:
        """Get drivers for a race – tries FastF1 first, falls back to OpenF1."""
        drivers = self.get_drivers_ff1(year, race)
        if drivers:
            return drivers
        if session_key is not None:
            return self.get_session_drivers(session_key)
        # Try to resolve session_key from race name
        sk = self.load_session(year, str(race))
        if sk is not None:
            return self.get_session_drivers(sk)
        return []

    def get_total_laps(
        self,
        year: int,
        race: str | int,
        session_key: Optional[int] = None,
    ) -> int:
        """Get total laps – tries FastF1 first, falls back to OpenF1."""
        total = self.get_total_laps_ff1(year, race)
        if total > 0:
            return total
        if session_key is not None:
            return self.get_race_total_laps(session_key)
        sk = self.load_session(year, str(race))
        if sk is not None:
            return self.get_race_total_laps(sk)
        return 0

    def get_laps(
        self,
        year: int,
        race: str | int,
        driver: str,
        session_key: Optional[int] = None,
        is_live: bool = False,
    ) -> pd.DataFrame:
        """Get race laps – tries FastF1 first, falls back to OpenF1."""
        df = self.get_race_laps_ff1(year, race, driver)
        if not df.empty:
            return df
        if session_key is not None:
            return self.get_race_laps(session_key, driver, is_live=is_live)
        sk = self.load_session(year, str(race))
        if sk is not None:
            return self.get_race_laps(sk, driver, is_live=is_live)
        return pd.DataFrame()

    def get_telemetry(
        self,
        year: int,
        race: str | int,
        driver: str,
        session_key: Optional[int] = None,
        is_live: bool = False,
    ) -> Tuple[Optional[Any], pd.DataFrame]:
        """Get fastest-lap telemetry – tries FastF1 first, falls back to OpenF1."""
        tel = self.get_telemetry_ff1(year, race, driver)
        if not tel.empty:
            return None, tel
        if session_key is not None:
            return self.get_fastest_lap_telemetry(session_key, driver, is_live=is_live)
        sk = self.load_session(year, str(race))
        if sk is not None:
            return self.get_fastest_lap_telemetry(sk, driver, is_live=is_live)
        return None, pd.DataFrame()

    # Alias for backward compatibility with app.py
    def get_race_laps_merged(
        self,
        year: int,
        race: str | int,
        driver: str,
        is_live: bool = False,
        session_key: Optional[int] = None,
    ) -> pd.DataFrame:
        """Alias for :meth:`get_laps` – kept for backward compatibility."""
        return self.get_laps(year, race, driver, session_key=session_key, is_live=is_live)
