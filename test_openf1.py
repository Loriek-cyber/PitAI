import data_handler
import requests

try:
    year = 2023
    grand_prix = "Italy"
    gp_mapping = {"Silverstone": "Great Britain", "Spa": "Belgium"}
    country = gp_mapping.get(grand_prix, grand_prix)
    url = "https://api.openf1.org/v1/sessions"
    params = {"year": year, "country_name": country}
    res = requests.get(url, params=params)
    sessions = res.json()
    q_session = [s for s in sessions if s.get('session_type') == 'Qualifying']
    print(q_session)
    if q_session:
        session_key = q_session[-1]['session_key']
        print("Q Session:", session_key)
        
        # Get driver number
        dr = data_handler.get_driver_number(session_key, 'VER')
        print("VER:", dr)
        
        # Get fastest lap
        res_l = requests.get("https://api.openf1.org/v1/laps", params={"session_key": session_key, "driver_number": dr})
        laps = [l for l in res_l.json() if l.get('lap_duration')]
        if laps:
            fastest = min(laps, key=lambda x: x['lap_duration'])
            print("Fastest Q lap duration:", fastest['lap_duration'])
except Exception as e:
    print(f"Error: {e}")
