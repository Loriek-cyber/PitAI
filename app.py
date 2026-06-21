import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="F1 AI Predictor", layout="wide", page_icon="🏎️")

st.title("F1 AI Predictor Dashboard")

# Sidebar Controls
st.sidebar.header("Race Selection")
mode = st.sidebar.radio("Data Mode", ["Historical", "Live"])
year = st.sidebar.selectbox("Season", list(range(2020, 2027)), index=4)

@st.cache_data
def fetch_races(selected_year):
    try:
        res = requests.get(f"{API_BASE_URL}/schedule/{selected_year}")
        if res.status_code == 200:
            return res.json().get("races", ["Monza"])
    except:
        pass
    return ["Monza"]

@st.cache_data
def fetch_max_laps(selected_year, selected_race):
    try:
        res = requests.get(f"{API_BASE_URL}/laps/{selected_year}/{selected_race}")
        if res.status_code == 200:
            return res.json().get("total_laps", 70)
    except:
        pass
    return 70

@st.cache_data
def fetch_drivers(selected_year, selected_race):
    try:
        res = requests.get(f"{API_BASE_URL}/drivers/{selected_year}/{selected_race}")
        if res.status_code == 200:
            return res.json().get("drivers", ["VER", "HAM"])
    except:
        pass
    return ["VER", "HAM"]

available_races = fetch_races(year)
race = st.sidebar.selectbox("Race Name", available_races)

# Main Layout
col1, col2 = st.columns((2, 1))

with col1:
    st.subheader("Race Win Probability")
    
    # Time Slider
    if mode == "Historical":
        max_laps = fetch_max_laps(year, race)
    else:
        max_laps = 50 # Example dynamic max
        
    selected_lap = st.slider("Race Moment (Lap)", min_value=0, max_value=max_laps, value=min(10, max_laps))
    
    # Fetch Probabilities
    if st.button("Calculate Probability"):
        with st.spinner("Analyzing AI inference..."):
            payload = {
                "year": year,
                "race": race,
                "lap": selected_lap,
                "is_live": mode == "Live"
            }
            try:
                response = requests.post(f"{API_BASE_URL}/probability", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    metrics = data.get("data", [])
                    
                    if metrics:
                        drivers = [m["Pilota"] for m in metrics]
                        probs = [m["Probabilità"] for m in metrics]
                        
                        # Chart
                        fig = px.bar(
                            x=drivers, 
                            y=probs,
                            labels={'x': 'Pilota', 'y': 'Probabilità di Vittoria'},
                            color=drivers,
                            title=f"Probabilità di Vittoria al Giro {selected_lap}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write("### Dettaglio e Metriche")
                        df_metrics = pd.DataFrame(metrics)
                        df_metrics["Probabilità"] = df_metrics["Probabilità"].apply(lambda x: f"{x:.1%}")
                        st.dataframe(df_metrics, hide_index=True, use_container_width=True)
                    else:
                        st.warning("Nessun dato disponibile per questo giro (es. giro 0 o dati non caricati).")
                else:
                    st.error(f"Error fetching data: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with col2:
    st.subheader("Head-to-Head Analysis")
    available_drivers = fetch_drivers(year, race)
    
    d1 = st.selectbox("Driver 1", available_drivers, index=0)
    d2_idx = 1 if len(available_drivers) > 1 else 0
    d2 = st.selectbox("Driver 2", available_drivers, index=d2_idx)
    
    if st.button("Compare Drivers"):
        with st.spinner("Loading telemetry..."):
            try:
                params = {"driver1": d1, "driver2": d2, "year": year, "race": race}
                res = requests.get(f"{API_BASE_URL}/head-to-head", params=params)
                
                if res.status_code == 200:
                    h2h_data = res.json()
                    st.metric(f"{d1} Success Rate", f"{h2h_data['driver1_success_rate']:.1%}")
                    st.metric(f"{d2} Success Rate", f"{h2h_data['driver2_success_rate']:.1%}")
                    
                    df = pd.DataFrame(h2h_data["lap_data"])
                    if not df.empty:
                        fig_delta = px.line(df, x="Lap", y="Delta", title=f"Lap Time Delta ({d1} vs {d2})")
                        fig_delta.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_delta, use_container_width=True)
                    else:
                        st.warning("No lap data available for comparison.")
                else:
                    st.error("Error fetching H2H data.")
            except Exception as e:
                st.error(f"Connection failed: {e}")
