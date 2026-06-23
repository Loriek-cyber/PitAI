import fastf1
import requests

# Monkey-patch su requests.Session.send per scavalcare il blocco
original_send = requests.Session.send
def custom_send(self, request, **kwargs):
    request.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    return original_send(self, request, **kwargs)

requests.Session.send = custom_send

fastf1.Cache.enable_cache('f1_cache')
try:
    session = fastf1.get_session(2023, 'Bahrain', 'R')
    session.load()
    laps = session.laps
    tel = laps.pick_driver('VER').pick_fastest().get_telemetry()
    print("Success: ", len(tel))
except Exception as e:
    print(f"Error: {e}")
