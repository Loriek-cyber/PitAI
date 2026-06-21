import asyncio
import pandas as pd
import fastf1 as ff1
from typing import Dict, Any, List

import os

class DataPipeline:
    def __init__(self, cache_dir: str = "./cache"):
        os.makedirs(cache_dir, exist_ok=True)
        ff1.Cache.enable_cache(cache_dir)
        self.openf1_ws_url = "wss://api.openf1.org/v1/live"
        
    async def get_historical_data(self, year: int, race: str, session: str = 'R') -> pd.DataFrame:
        """Fetches historical telemetry, lap times, and strategies using FastF1."""
        loop = asyncio.get_running_loop()
        f1_session = await loop.run_in_executor(None, self._load_session, year, race, session)
        
        laps_data = f1_session.laps
        telemetry_data = f1_session.pos_data
        
        return self._normalize_metrics(laps_data, telemetry_data)

    def _load_session(self, year: int, race: str, session: str):
        s = ff1.get_session(year, race, session)
        s.load()
        return s

    def get_schedule(self, year: int) -> List[str]:
        schedule = ff1.get_event_schedule(year)
        races = schedule[schedule['EventFormat'] != 'testing']['EventName'].tolist()
        return races

    async def get_total_laps(self, year: int, race: str) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_laps, year, race)

    def _fetch_laps(self, year: int, race: str) -> int:
        try:
            s = ff1.get_session(year, race, 'R')
            s.load(telemetry=False, weather=False, messages=False)
            return int(s.total_laps) if s.total_laps else 70
        except Exception:
            return 70

    async def get_drivers(self, year: int, race: str) -> List[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_drivers, year, race)

    def _fetch_drivers(self, year: int, race: str) -> List[str]:
        try:
            s = ff1.get_session(year, race, 'R')
            s.load(telemetry=False, weather=False, messages=False)
            return s.results['Abbreviation'].dropna().tolist()
        except Exception:
            return ["VER", "HAM", "NOR", "LEC", "SAI", "RUS", "PIA", "PER", "ALO", "STR"]

    async def get_lap_metrics(self, year: int, race: str, lap_number: int) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_lap_metrics, year, race, lap_number)

    def _fetch_lap_metrics(self, year: int, race: str, lap_number: int) -> pd.DataFrame:
        try:
            s = ff1.get_session(year, race, 'R')
            s.load(telemetry=False, weather=False, messages=False)
            laps = s.laps
            lap_data = laps[laps['LapNumber'] == lap_number]
            
            if lap_data.empty:
                return pd.DataFrame()
                
            total_laps = int(s.total_laps) if s.total_laps else 70
            
            df = lap_data[['Driver', 'LapTime', 'Compound', 'TyreLife', 'SpeedST', 'TrackStatus', 'Position', 'Stint']].copy()
            df['LapTime_s'] = df['LapTime'].dt.total_seconds()
            df['TotalLaps'] = total_laps
            df['LapNumber'] = lap_number
            return df
        except Exception:
            return pd.DataFrame()

    async def connect_live_stream(self):
        """Placeholder for OpenF1 API real-time websocket connection."""
        # import websockets
        # async with websockets.connect(self.openf1_ws_url) as websocket:
        #     while True:
        #         message = await websocket.recv()
        #         self._process_live_data(message)
        pass

    def _normalize_metrics(self, laps: pd.DataFrame, telemetry: Dict[str, Any]) -> pd.DataFrame:
        """Normalizes lap times, tire degradation, and sector times into standard format."""
        if laps.empty:
            return pd.DataFrame()
        
        normalized_df = laps[['Driver', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife']].copy()
        normalized_df['LapTime_s'] = normalized_df['LapTime'].dt.total_seconds()
        normalized_df = normalized_df.fillna(0)
        return normalized_df

    async def fetch_driver_head_to_head(self, driver1: str, driver2: str, year: int, race: str) -> pd.DataFrame:
        """Fetches and compares lap deltas between two specific drivers."""
        data = await self.get_historical_data(year, race)
        d1_data = data[data['Driver'] == driver1].reset_index(drop=True)
        d2_data = data[data['Driver'] == driver2].reset_index(drop=True)
        
        comparison = pd.DataFrame({
            'Lap': d1_data.index + 1,
            f'{driver1}_Time': d1_data['LapTime_s'],
            f'{driver2}_Time': d2_data['LapTime_s'],
            'Delta': d1_data['LapTime_s'] - d2_data['LapTime_s']
        })
        return comparison
