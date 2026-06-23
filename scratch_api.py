import requests
import time

session_key = 9153 # Monza 2023 Qualy as example, let's use a race instead
res = requests.get("https://api.openf1.org/v1/sessions?year=2023&country_name=Italy")
sessions = [s for s in res.json() if s['session_type'] == 'Race']
r_session = sessions[-1]['session_key']
print("Race session:", r_session)

# 1. Test Track Status
t0 = time.time()
res_ts = requests.get(f"https://api.openf1.org/v1/track_status?session_key={r_session}")
print("Track status fetched in", time.time()-t0, "s. Len:", len(res_ts.json()))

# 2. Test Car Data for a whole race for driver 1 (VER)
t0 = time.time()
res_cd = requests.get(f"https://api.openf1.org/v1/car_data?session_key={r_session}&driver_number=1")
car_data = res_cd.json()
print("Car data fetched in", time.time()-t0, "s. Len:", len(car_data))
if car_data:
    print(car_data[0])

