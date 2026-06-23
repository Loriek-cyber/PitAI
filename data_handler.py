import requests
import pandas as pd
import numpy as np
import os
import pickle
import time

CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

# Flag globale per tracciare se è stata effettuata una chiamata di rete
NETWORK_HIT = False

class MockResponse:
    def __init__(self, json_data):
        self._json = json_data
        self.status_code = 200
    def json(self):
        return self._json

def _requests_get_with_retry(url, params=None, max_retries=3):
    global NETWORK_HIT
    NETWORK_HIT = True
    for i in range(max_retries):
        res = requests.get(url, params=params)
        if res.status_code == 429 or res.status_code >= 500:
            time.sleep((i + 1) * 3) # Backoff
            continue
        if res.status_code == 404:
            return MockResponse([])
        if res.status_code != 200:
            raise ConnectionError(f"Errore OpenF1 API: {res.status_code} su {url}")
        return res
    raise ConnectionError("Limite richieste OpenF1 API raggiunto (429 Too Many Requests o Errori Server) dopo multipli tentativi.")

def _cached_api_call(cache_name: str, fetch_func):
    """Esegue il fetch_func e salva il risultato su disco per i dati offline."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    data = fetch_func()
    if data is not None:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass
    return data

def get_available_races(year: int):
    def _fetch():
        url = f"https://api.openf1.org/v1/sessions?year={year}"
        res = _requests_get_with_retry(url)
        sessions = res.json()
        if isinstance(sessions, dict):
            sessions = []
        races = []
        for s in sessions:
            if s.get('session_type') == 'Race':
                races.append(s.get('country_name'))
        return list(dict.fromkeys(races))
    return _cached_api_call(f"races_{year}", _fetch)

def get_session_drivers(session_key: int):
    def _fetch():
        url = f"https://api.openf1.org/v1/drivers?session_key={session_key}"
        res = _requests_get_with_retry(url)
        drivers = res.json()
        if isinstance(drivers, dict):
            drivers = []
        acronyms = [d['name_acronym'] for d in drivers if 'name_acronym' in d]
        return list(dict.fromkeys(acronyms))
    return _cached_api_call(f"drivers_{session_key}", _fetch)

def get_driver_number(session_key: int, driver_acronym: str):
    def _fetch():
        url = f"https://api.openf1.org/v1/drivers?session_key={session_key}&name_acronym={driver_acronym}"
        res = _requests_get_with_retry(url)
        data = res.json()
        if data and isinstance(data, list):
            return data[0]['driver_number']
        return 1 # Fallback
    return _cached_api_call(f"drvno_{session_key}_{driver_acronym}", _fetch)

def get_race_total_laps(session_key: int):
    def _fetch():
        res = _requests_get_with_retry("https://api.openf1.org/v1/laps", params={"session_key": session_key})
        data = res.json()
        if isinstance(data, dict):
            data = []
        if data:
            return max([l.get('lap_number', 0) for l in data])
        return 50 # Fallback
    return _cached_api_call(f"totlaps_{session_key}", _fetch)

def get_qualy_fastest_lap(qualy_session_key: int, driver_acronym: str):
    def _fetch():
        dr = get_driver_number(qualy_session_key, driver_acronym)
        res = _requests_get_with_retry("https://api.openf1.org/v1/laps", params={"session_key": qualy_session_key, "driver_number": dr})
        data = res.json()
        if isinstance(data, dict):
            data = []
        laps = [l for l in data if l.get('lap_duration')]
        if laps:
            fastest = min(laps, key=lambda x: x['lap_duration'])
            return fastest['lap_duration']
        return None
    return _cached_api_call(f"qlap_{qualy_session_key}_{driver_acronym}", _fetch)

def load_session(year: int, grand_prix: str, session_type: str = 'R'):
    def _fetch():
        gp_mapping = {
            "Silverstone": "Great Britain",
            "Spa": "Belgium"
        }
        country = gp_mapping.get(grand_prix, grand_prix)
        
        url = "https://api.openf1.org/v1/sessions"
        params = {"year": year, "country_name": country}
        res = _requests_get_with_retry(url, params=params)
        
        data = res.json()
        if isinstance(data, dict):
            data = []
            
        sessions = [s for s in data if s.get('session_type') == ('Race' if session_type == 'R' else 'Qualifying')]
        if not sessions:
            res = _requests_get_with_retry(f"https://api.openf1.org/v1/sessions", params={"year": year})
            all_sessions = res.json()
            if isinstance(all_sessions, dict):
                all_sessions = []
            for s in all_sessions:
                if s.get('session_type') == ('Race' if session_type == 'R' else 'Qualifying') and country.lower() in s.get('country_name', '').lower():
                    return s['session_key']
            raise ValueError(f"Sessione non trovata per {year} {grand_prix}")
            
        return sessions[-1]['session_key']
    return _cached_api_call(f"session_{year}_{grand_prix}_{session_type}", _fetch)

def get_race_laps(session_key, driver: str, is_live: bool = False):
    cache_path = os.path.join(CACHE_DIR, f"laps_{session_key}_{driver}.pkl")
    if not is_live and os.path.exists(cache_path):
        try:
            return pd.read_pickle(cache_path)
        except:
            pass

    driver_no = get_driver_number(session_key, driver)
    
    res = _requests_get_with_retry("https://api.openf1.org/v1/laps", params={"session_key": session_key, "driver_number": driver_no})
        
    laps_data = res.json()
    if isinstance(laps_data, dict):
        laps_data = []
    df_laps = pd.DataFrame(laps_data)
    
    res_stints = _requests_get_with_retry("https://api.openf1.org/v1/stints", params={"session_key": session_key, "driver_number": driver_no})
    stints_data = res_stints.json()
    if isinstance(stints_data, dict):
        stints_data = []

    res_car = _requests_get_with_retry("https://api.openf1.org/v1/car_data", params={"session_key": session_key, "driver_number": driver_no})
    car_data = res_car.json()
    if isinstance(car_data, dict):
        car_data = []
    if car_data:
        df_car = pd.DataFrame(car_data)
        df_car['date'] = pd.to_datetime(df_car['date'], format='ISO8601')
    else:
        df_car = pd.DataFrame(columns=['date', 'throttle', 'brake'])

    res_ts = _requests_get_with_retry("https://api.openf1.org/v1/track_status", params={"session_key": session_key})
    ts_data = res_ts.json()
    if isinstance(ts_data, dict):
        ts_data = []
    if ts_data:
        df_ts = pd.DataFrame(ts_data)
        df_ts['date'] = pd.to_datetime(df_ts['date'], format='ISO8601')
        df_ts = df_ts.sort_values('date')
    else:
        df_ts = pd.DataFrame(columns=['date', 'status'])
    
    out = []
    tyre_age = 1
    current_stint = 1
    current_compound = 'SOFT'
    stint_idx = 0
    
    for _, row in df_laps.iterrows():
        lap_num = row.get('lap_number', 0)
        
        if stint_idx < len(stints_data):
            st = stints_data[stint_idx]
            
            lap_start = st.get('lap_start')
            lap_start = lap_start if lap_start is not None else 0
            
            lap_end = st.get('lap_end')
            lap_end = lap_end if lap_end is not None else 999
            
            tyre_start = st.get('tyre_age_at_start')
            tyre_start = tyre_start if tyre_start is not None else 0
            
            if lap_num >= lap_start and lap_num <= lap_end:
                current_stint = st.get('stint_number', 1)
                current_compound = st.get('compound', 'UNKNOWN')
                tyre_age = tyre_start + (lap_num - lap_start)
            elif lap_num > lap_end:
                stint_idx += 1
                if stint_idx < len(stints_data):
                    st = stints_data[stint_idx]
                    
                    lap_start = st.get('lap_start')
                    lap_start = lap_start if lap_start is not None else 0
                    
                    tyre_start = st.get('tyre_age_at_start')
                    tyre_start = tyre_start if tyre_start is not None else 0

                    current_stint = st.get('stint_number', 1)
                    current_compound = st.get('compound', 'UNKNOWN')
                    tyre_age = tyre_start + (lap_num - lap_start)

        ld = row.get('lap_duration')
        ds = row.get('date_start')
        
        avg_thr = 0.0
        avg_brk = 0.0
        is_sc = 0
        
        if pd.notna(ld) and ds:
            ds_dt = pd.to_datetime(ds, format='ISO8601')
            de_dt = ds_dt + pd.to_timedelta(ld, unit='s')
            
            if not df_car.empty:
                lap_car = df_car[(df_car['date'] >= ds_dt) & (df_car['date'] <= de_dt)]
                if not lap_car.empty:
                    avg_thr = lap_car['throttle'].mean()
                    avg_brk = lap_car['brake'].mean()
                    
            if not df_ts.empty:
                prev_statuses = df_ts[df_ts['date'] <= de_dt]
                if not prev_statuses.empty:
                    last_status = prev_statuses.iloc[-1]['status']
                    if str(last_status) in ['4', '5']:
                        is_sc = 1

        out.append({
            'LapNumber': lap_num,
            'LapTime': pd.to_timedelta(ld, unit='s') if ld else pd.NaT,
            'Sector1Time': pd.to_timedelta(row.get('duration_sector_1'), unit='s') if row.get('duration_sector_1') else pd.NaT,
            'Sector2Time': pd.to_timedelta(row.get('duration_sector_2'), unit='s') if row.get('duration_sector_2') else pd.NaT,
            'Sector3Time': pd.to_timedelta(row.get('duration_sector_3'), unit='s') if row.get('duration_sector_3') else pd.NaT,
            'TyreLife': tyre_age,
            'Compound': current_compound,
            'Stint': current_stint,
            'IsAccurate': pd.notna(ld),
            'date_start': row.get('date_start'),
            'AvgThrottle': avg_thr,
            'AvgBrake': avg_brk,
            'IsSC': is_sc
        })
        
    df_out = pd.DataFrame(out)
    if not is_live and not df_out.empty:
        df_out.to_pickle(cache_path)
        
    return df_out

def get_fastest_lap_telemetry(session_key, driver: str, is_live: bool = False):
    cache_path = os.path.join(CACHE_DIR, f"tel_{session_key}_{driver}.pkl")
    if not is_live and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass

    driver_no = get_driver_number(session_key, driver)
    
    res = _requests_get_with_retry("https://api.openf1.org/v1/laps", params={"session_key": session_key, "driver_number": driver_no})
        
    data = res.json()
    if isinstance(data, dict):
        data = []
    laps_data = [l for l in data if l.get('lap_duration')]
    
    if not laps_data:
        return None, pd.DataFrame()
        
    fastest = min(laps_data, key=lambda x: x['lap_duration'])
    
    start_ts = pd.to_datetime(fastest['date_start'], format='ISO8601')
    end_ts = start_ts + pd.to_timedelta(fastest['lap_duration'], unit='s')
    
    start_ts_str = start_ts.isoformat().replace('+', '%2B')
    end_ts_str = end_ts.isoformat().replace('+', '%2B')
    url = f"https://api.openf1.org/v1/car_data?session_key={session_key}&driver_number={driver_no}&date>={start_ts_str}&date<={end_ts_str}"
    
    res_car = _requests_get_with_retry(url)
        
    car_data = res_car.json()
    
    if isinstance(car_data, dict):
        car_data = []
    df_car = pd.DataFrame(car_data)
    
    if df_car.empty:
        return fastest, pd.DataFrame()
        
    df_car['date'] = pd.to_datetime(df_car['date'], format='ISO8601')
    df_car = df_car.sort_values('date')
    
    df_car['Time_diff'] = df_car['date'].diff().dt.total_seconds().fillna(0)
    df_car['Speed_ms'] = df_car['speed'] / 3.6
    df_car['Distance'] = (df_car['Speed_ms'] * df_car['Time_diff']).cumsum()
    
    df_car = df_car.rename(columns={'speed': 'Speed'})
    
    if not is_live and not df_car.empty:
        with open(cache_path, 'wb') as f:
            pickle.dump((fastest, df_car), f)
            
    return fastest, df_car

