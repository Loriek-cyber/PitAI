"""
test_data_core.py – Pytest tests for the DataCore unified data layer.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_core import DataCore, MockResponse, GP_MAPPING


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dc(tmp_path):
    """Provide a DataCore instance with temporary cache dirs."""
    cache_dir = str(tmp_path / "cache_data")
    ff1_cache_dir = str(tmp_path / "f1_local_cache")
    return DataCore(cache_dir=cache_dir, ff1_cache_dir=ff1_cache_dir)


# ---------------------------------------------------------------------------
# MockResponse
# ---------------------------------------------------------------------------

class TestMockResponse:
    def test_status_code(self):
        r = MockResponse()
        assert r.status_code == 404

    def test_json_returns_list(self):
        r = MockResponse()
        assert r.json() == []


# ---------------------------------------------------------------------------
# GP_MAPPING
# ---------------------------------------------------------------------------

class TestGPMapping:
    def test_silverstone(self):
        assert GP_MAPPING["Silverstone"] == "Great Britain"

    def test_spa(self):
        assert GP_MAPPING["Spa"] == "Belgium"

    def test_monza(self):
        assert GP_MAPPING["Monza"] == "Italy"


# ---------------------------------------------------------------------------
# DataCore initialisation
# ---------------------------------------------------------------------------

class TestDataCoreInit:
    def test_cache_dirs_created(self, dc, tmp_path):
        assert os.path.isdir(str(tmp_path / "cache_data"))
        assert os.path.isdir(str(tmp_path / "f1_local_cache"))

    def test_network_hit_initially_false(self, dc):
        assert dc.network_hit is False


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @patch("data_core.requests.get")
    def test_returns_on_200(self, mock_get, dc):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        result = dc._requests_get_with_retry("https://example.com/test")
        assert result.status_code == 200
        assert dc.network_hit is True

    @patch("data_core.requests.get")
    def test_returns_mock_on_persistent_failure(self, mock_get, dc):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        result = dc._requests_get_with_retry("https://example.com/test", max_retries=1)
        assert isinstance(result, MockResponse)

    @patch("data_core.requests.get")
    def test_returns_mock_on_non_retryable_status(self, mock_get, dc):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp

        result = dc._requests_get_with_retry("https://example.com/test")
        assert isinstance(result, MockResponse)


# ---------------------------------------------------------------------------
# Cached API call
# ---------------------------------------------------------------------------

class TestCachedApiCall:
    def test_caches_and_returns(self, dc):
        call_count = 0

        def fetcher():
            nonlocal call_count
            call_count += 1
            return {"data": 42}

        result1 = dc._cached_api_call("test_cache", fetcher)
        result2 = dc._cached_api_call("test_cache", fetcher)

        assert result1 == {"data": 42}
        assert result2 == {"data": 42}
        assert call_count == 1  # second call served from cache


# ---------------------------------------------------------------------------
# OpenF1: get_available_races
# ---------------------------------------------------------------------------

class TestGetAvailableRaces:
    @patch("data_core.requests.get")
    def test_returns_races(self, mock_get, dc):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"session_key": 1, "meeting_name": "Italian Grand Prix"},
            {"session_key": 2, "meeting_name": "British Grand Prix"},
        ]
        mock_get.return_value = mock_resp

        races = dc.get_available_races(2024)
        assert len(races) == 2
        assert races[0]["meeting_name"] == "Italian Grand Prix"


# ---------------------------------------------------------------------------
# OpenF1: get_session_drivers
# ---------------------------------------------------------------------------

class TestGetSessionDrivers:
    @patch("data_core.requests.get")
    def test_returns_unique_drivers(self, mock_get, dc):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"name_acronym": "VER"},
            {"name_acronym": "HAM"},
            {"name_acronym": "VER"},  # duplicate
        ]
        mock_get.return_value = mock_resp

        drivers = dc.get_session_drivers(9999)
        assert drivers == ["VER", "HAM"]


# ---------------------------------------------------------------------------
# OpenF1: get_driver_number
# ---------------------------------------------------------------------------

class TestGetDriverNumber:
    @patch("data_core.requests.get")
    def test_returns_number(self, mock_get, dc):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"driver_number": 1}]
        mock_get.return_value = mock_resp

        assert dc.get_driver_number(9999, "VER") == 1

    @patch("data_core.requests.get")
    def test_returns_none_when_not_found(self, mock_get, dc):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        assert dc.get_driver_number(9999, "ZZZ") is None


# ---------------------------------------------------------------------------
# OpenF1: load_session
# ---------------------------------------------------------------------------

class TestLoadSession:
    @patch("data_core.requests.get")
    def test_finds_session_by_meeting_name(self, mock_get, dc):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "session_key": 9574,
                "meeting_name": "Italian Grand Prix",
                "country_name": "Italy",
                "location": "Monza",
            },
        ]
        mock_get.return_value = mock_resp

        sk = dc.load_session(2024, "Monza")
        assert sk == 9574

    @patch("data_core.requests.get")
    def test_returns_none_when_not_found(self, mock_get, dc):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        assert dc.load_session(2024, "Nonexistent") is None

    @patch("data_core.requests.get")
    def test_gp_mapping_used(self, mock_get, dc):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "session_key": 1234,
                "meeting_name": "British Grand Prix",
                "country_name": "Great Britain",
                "location": "Silverstone",
            },
        ]
        mock_get.return_value = mock_resp

        # "Silverstone" maps to "Great Britain" via GP_MAPPING
        sk = dc.load_session(2024, "Silverstone")
        assert sk == 1234


# ---------------------------------------------------------------------------
# OpenF1: get_race_total_laps
# ---------------------------------------------------------------------------

class TestGetRaceTotalLaps:
    @patch("data_core.requests.get")
    def test_returns_max_lap(self, mock_get, dc):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"lap_number": 1},
            {"lap_number": 53},
            {"lap_number": 27},
        ]
        mock_get.return_value = mock_resp

        assert dc.get_race_total_laps(9999) == 53


# ---------------------------------------------------------------------------
# OpenF1: get_race_laps (core function)
# ---------------------------------------------------------------------------

class TestGetRaceLaps:
    @patch("data_core.requests.get")
    def test_returns_enriched_dataframe(self, mock_get, dc):
        """Verify the full get_race_laps pipeline with stints, car_data, track_status."""
        # We need to mock 5 sequential API calls:
        # 1) get_driver_number, 2) laps, 3) stints, 4) car_data, 5) track_status
        driver_resp = MagicMock()
        driver_resp.status_code = 200
        driver_resp.json.return_value = [{"driver_number": 1}]

        laps_resp = MagicMock()
        laps_resp.status_code = 200
        laps_resp.json.return_value = [
            {
                "lap_number": 1,
                "lap_duration": 95.5,
                "date_start": "2024-09-01T14:00:00+00:00",
                "duration_sector_1": 30.0,
                "duration_sector_2": 35.0,
                "duration_sector_3": 30.5,
            },
        ]

        stints_resp = MagicMock()
        stints_resp.status_code = 200
        stints_resp.json.return_value = [
            {
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": 20,
                "compound": "SOFT",
                "tyre_age_at_start": 0,
            },
        ]

        car_resp = MagicMock()
        car_resp.status_code = 200
        car_resp.json.return_value = [
            {"date": "2024-09-01T14:00:10+00:00", "throttle": 80, "brake": 10},
            {"date": "2024-09-01T14:00:50+00:00", "throttle": 90, "brake": 5},
        ]

        ts_resp = MagicMock()
        ts_resp.status_code = 200
        ts_resp.json.return_value = [
            {"date": "2024-09-01T13:59:00+00:00", "status": "1"},
        ]

        mock_get.side_effect = [driver_resp, laps_resp, stints_resp, car_resp, ts_resp]

        df = dc.get_race_laps(9999, "VER", is_live=True)
        assert not df.empty
        assert "LapNumber" in df.columns
        assert "Compound" in df.columns
        assert "AvgThrottle" in df.columns
        assert "AvgBrake" in df.columns
        assert "IsSC" in df.columns
        assert "TyreLife" in df.columns
        assert df.iloc[0]["Compound"] == "SOFT"
        assert df.iloc[0]["IsSC"] == 0


# ---------------------------------------------------------------------------
# OpenF1: get_fastest_lap_telemetry
# ---------------------------------------------------------------------------

class TestGetFastestLapTelemetry:
    @patch("data_core.requests.get")
    def test_returns_telemetry_with_distance(self, mock_get, dc):
        driver_resp = MagicMock()
        driver_resp.status_code = 200
        driver_resp.json.return_value = [{"driver_number": 1}]

        laps_resp = MagicMock()
        laps_resp.status_code = 200
        laps_resp.json.return_value = [
            {"lap_duration": 90.0, "date_start": "2024-09-01T14:00:00+00:00"},
            {"lap_duration": 88.5, "date_start": "2024-09-01T14:02:00+00:00"},
        ]

        car_resp = MagicMock()
        car_resp.status_code = 200
        car_resp.json.return_value = [
            {"date": "2024-09-01T14:02:00+00:00", "speed": 200},
            {"date": "2024-09-01T14:02:30+00:00", "speed": 250},
            {"date": "2024-09-01T14:03:00+00:00", "speed": 300},
        ]

        mock_get.side_effect = [driver_resp, laps_resp, car_resp]

        fastest, tel_df = dc.get_fastest_lap_telemetry(9999, "VER", is_live=True)
        assert fastest is not None
        assert fastest["lap_duration"] == 88.5
        assert not tel_df.empty
        assert "Distance" in tel_df.columns
        assert "Speed" in tel_df.columns


# ---------------------------------------------------------------------------
# FastF1 methods (mocked)
# ---------------------------------------------------------------------------

class TestFastF1Methods:
    @patch("data_core.ff1.get_event_schedule")
    def test_get_schedule(self, mock_sched, dc):
        mock_sched.return_value = pd.DataFrame({"EventName": ["GP1", "GP2"]})
        sched = dc.get_schedule(2024)
        assert len(sched) == 2

    @patch("data_core.ff1.get_session")
    def test_get_drivers_ff1(self, mock_session, dc):
        mock_sess = MagicMock()
        mock_sess.results = pd.DataFrame({"Abbreviation": ["VER", "HAM", "LEC"]})
        mock_session.return_value = mock_sess
        drivers = dc.get_drivers_ff1(2024, "Monza")
        assert drivers == ["VER", "HAM", "LEC"]

    @patch("data_core.ff1.get_session")
    def test_get_drivers_ff1_failure(self, mock_session, dc):
        mock_session.side_effect = Exception("Network error")
        drivers = dc.get_drivers_ff1(2024, "Monza")
        assert drivers == []

    @patch("data_core.ff1.get_session")
    def test_get_total_laps_ff1(self, mock_session, dc):
        mock_sess = MagicMock()
        mock_sess.laps = pd.DataFrame({"LapNumber": [1, 2, 3, 53]})
        mock_session.return_value = mock_sess
        total = dc.get_total_laps_ff1(2024, "Monza")
        assert total == 53

    @patch("data_core.ff1.get_session")
    def test_get_total_laps_ff1_failure(self, mock_session, dc):
        mock_session.side_effect = Exception("Error")
        total = dc.get_total_laps_ff1(2024, "Monza")
        assert total == 0


# ---------------------------------------------------------------------------
# Unified convenience methods
# ---------------------------------------------------------------------------

class TestUnifiedMethods:
    @patch("data_core.ff1.get_session")
    def test_get_drivers_uses_ff1_first(self, mock_session, dc):
        mock_sess = MagicMock()
        mock_sess.results = pd.DataFrame({"Abbreviation": ["VER", "HAM"]})
        mock_session.return_value = mock_sess

        drivers = dc.get_drivers(2024, "Monza")
        assert drivers == ["VER", "HAM"]

    @patch("data_core.ff1.get_session")
    @patch("data_core.requests.get")
    def test_get_drivers_falls_back_to_openf1(self, mock_get, mock_session, dc):
        # FastF1 fails
        mock_session.side_effect = Exception("FastF1 error")

        # OpenF1 succeeds (two calls: load_session + get_session_drivers)
        session_resp = MagicMock()
        session_resp.status_code = 200
        session_resp.json.return_value = [
            {"session_key": 1234, "meeting_name": "Italian GP", "country_name": "Italy", "location": "Monza"},
        ]
        drivers_resp = MagicMock()
        drivers_resp.status_code = 200
        drivers_resp.json.return_value = [
            {"name_acronym": "VER"},
            {"name_acronym": "LEC"},
        ]
        mock_get.side_effect = [session_resp, drivers_resp]

        drivers = dc.get_drivers(2024, "Monza")
        assert "VER" in drivers
