import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import data_handler
import model_engine
import time

# Configurazione base dell'app
st.set_page_config(page_title="F1 Telemetry & Pace AI", layout="wide", initial_sidebar_state="expanded")

# Stile Premium / Dynamic Design
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(33, 33, 45) 0%, rgb(18, 18, 24) 90%);
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #4da6ff;
        font-family: 'Inter', sans-serif;
        text-shadow: 0px 4px 10px rgba(77,166,255,0.4);
    }
    .stSidebar {
        background-color: rgba(22, 27, 34, 0.6) !important;
        backdrop-filter: blur(15px);
    }
    .css-1d391kg {
        background: transparent;
    }
    div[data-testid="stDataFrame"] {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="stDataFrame"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(77, 166, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

st.title("🏎️ F1 Telemetry Analysis & Race Pace Prediction")

# Barra laterale per i controlli
st.sidebar.header("Impostazioni Sessione")
year = st.sidebar.selectbox("Stagione", [2026, 2025, 2024, 2023, 2022], index=0)

@st.cache_data
def get_races_for_year(y):
    try:
        return data_handler.get_available_races(y)
    except:
        return ["Italy", "Great Britain", "Bahrain", "Monaco", "Belgium"]

available_races = get_races_for_year(year)
race = st.sidebar.selectbox("Gran Premio", available_races, index=0)

@st.cache_data
def get_drivers_for_session(y, r):
    session_key = data_handler.load_session(y, r, 'R')
    return data_handler.get_session_drivers(session_key)

@st.cache_data
def get_race_meta(y, r):
    try:
        session_key = data_handler.load_session(y, r, 'R')
        return data_handler.get_race_total_laps(session_key)
    except:
        return 50

available_drivers = get_drivers_for_session(year, race)
tot_laps = get_race_meta(year, race)

st.sidebar.markdown("---")
telemetry_drivers = st.sidebar.multiselect(
    "Piloti per Speed Trace", 
    available_drivers, 
    default=available_drivers[:3] if len(available_drivers) >= 3 else available_drivers
)

is_live_mode = st.sidebar.checkbox("🔴 Modalità Gara Live (Ignora Cache Storica)", value=False)
st.sidebar.markdown("---")
laps_to_load = st.sidebar.number_input("Giri da caricare (Template Gara)", min_value=1, max_value=tot_laps, value=tot_laps)
starting_lap = st.sidebar.number_input("Giro di Partenza (0 = Qualifica)", min_value=0, max_value=laps_to_load, value=0)
num_laps_predict = st.sidebar.number_input("Numero di Giri da prevedere", min_value=1, value=tot_laps - starting_lap if (tot_laps - starting_lap) > 0 else 5)

def get_driver_data(y, r, d, is_live):
    if is_live:
        session = data_handler.load_session(y, r, 'R')
        lap_f, tel = data_handler.get_fastest_lap_telemetry(session, d, is_live=True)
        laps_d = data_handler.get_race_laps(session, d, is_live=True)
        return tel, laps_d
    else:
        return _get_driver_data_cached(y, r, d)

@st.cache_data(show_spinner=False)
def _get_driver_data_cached(y, r, d):
    session = data_handler.load_session(y, r, 'R')
    lap_f, tel = data_handler.get_fastest_lap_telemetry(session, d, is_live=False)
    laps_d = data_handler.get_race_laps(session, d, is_live=False)
    return tel, laps_d

@st.cache_data(show_spinner=False)
def get_prediction_for_driver(df_laps, laps_to_load, start_lap, num_laps, drv_acronym, y, r):
    if df_laps is None or df_laps.empty:
        return None
        
    df_prep, le = model_engine.prepare_features(df_laps)
    df_train = df_prep[df_prep['LapNumber'] <= laps_to_load]
    
    if df_train.empty:
        return None
        
    model = model_engine.train_pace_model(df_train)
    
    if start_lap == 0:
        try:
            q_session = data_handler.load_session(y, r, 'Q')
            q_lap = data_handler.get_qualy_fastest_lap(q_session, drv_acronym)
        except:
            q_lap = None
        
        c_lap = 0
        c_tyre = 0
        c_comp = df_train.iloc[0]['Compound_encoded'] if not df_train.empty else 0
        c_stint = 1
        
        c_laptime = q_lap if q_lap else (df_train.iloc[0]['LapTime_sec'] if not df_train.empty else 90.0)
        avg_thr = df_train['AvgThrottle'].mean() if not df_train.empty else 0.0
        avg_brk = df_train['AvgBrake'].mean() if not df_train.empty else 0.0
    else:
        subset = df_train[df_train['LapNumber'] <= start_lap]
        if not subset.empty:
            last_known = subset.iloc[-1]
            c_lap = last_known['LapNumber']
            c_tyre = last_known['TyreLife']
            c_comp = last_known['Compound_encoded']
            c_stint = last_known['Stint']
            c_laptime = last_known['LapTime_sec']
            avg_thr = subset['AvgThrottle'].mean()
            avg_brk = subset['AvgBrake'].mean()
        else:
            c_lap = start_lap
            c_tyre = start_lap
            c_comp = 0
            c_stint = 1
            c_laptime = 90.0
            avg_thr = 0.0
            avg_brk = 0.0
    
    preds = model_engine.predict_future_pace(
        model, current_lap=c_lap, current_tyre_life=c_tyre,
        current_compound_enc=c_comp, current_stint=c_stint, 
        current_laptime=c_laptime, avg_throttle=avg_thr, avg_brake=avg_brk,
        num_laps=num_laps
    )
    
    if start_lap == 0 and q_lap:
        q_row = pd.DataFrame([{
            'LapNumber': 0,
            'TyreLife': 0,
            'Compound_encoded': c_comp,
            'Stint': 1,
            'Predicted_LapTime_sec': q_lap
        }])
        preds = pd.concat([q_row, preds], ignore_index=True)
        
    return preds

try:
    all_tels = {}
    all_laps = {}
    
    st.markdown("---")
    st.subheader(f"🌐 Raccolta Dati Roster Completo ({len(available_drivers)} Piloti)")
    
    progress_bar = st.progress(0, text="Avvio download dati da OpenF1 API...")
    
    for idx, d in enumerate(available_drivers):
        progress_bar.progress((idx) / len(available_drivers), text=f"Scaricamento dati: {d} ({idx+1}/{len(available_drivers)}) - Include Telemetria e Safety Car...")
        
        t0 = time.time()
        try:
            tel, laps = get_driver_data(year, race, d, is_live=is_live_mode)
            all_tels[d] = tel
            all_laps[d] = laps
        except Exception as e:
            all_tels[d] = pd.DataFrame()
            all_laps[d] = pd.DataFrame()
            
        # Se la chiamata ha richiesto tempo (cache miss), riposiamo per evitare il 429
        if time.time() - t0 > 0.5:
            time.sleep(2)
            
    progress_bar.empty()
    st.success("Tutti i dati dell'intero schieramento sono stati elaborati!")
    
    # SEZIONE 1: SPEED TRACE INTERATTIVA
    st.markdown("---")
    st.subheader(f"⚡ Speed Trace Interattiva (Giro più Veloce)")
    st.markdown("Confronto della velocità lungo il tracciato. Aggiungi o rimuovi piloti dalla barra laterale sinistra.")
    
    if len(telemetry_drivers) > 0:
        fig_speed = go.Figure()
        colors = px.colors.qualitative.Alphabet
        
        for idx, d in enumerate(telemetry_drivers):
            if d in all_tels and not all_tels[d].empty:
                dist, speed = model_engine.interpolate_telemetry(all_tels[d])
                c = colors[idx % len(colors)]
                fig_speed.add_trace(go.Scatter(x=dist, y=speed, mode='lines', name=d, line=dict(color=c, width=2.5)))
                
        fig_speed.update_layout(
            xaxis_title="Distanza Percorsa (metri)",
            yaxis_title="Velocità (km/h)",
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_speed, width='stretch')
    else:
        st.info("Seleziona almeno un pilota dalla barra laterale per mostrare il grafico della velocità.")

    # SEZIONE 2: XGBOOST RACE PACE PREDICTION
    st.markdown("---")
    st.subheader("🔮 Previsione Passo Gara Globale (Modello XGBoost)")
    st.markdown("L'IA ha addestrato un modello individuale per ogni pilota calcolando Degrado Gomma, Pressione Pedali e Autoregressione. Ecco il passo gara di tutta la griglia.")
    
    all_preds = {}
    with st.spinner("Addestramento di 20 modelli XGBoost e generazione predizioni..."):
        for d in available_drivers:
            p = get_prediction_for_driver(all_laps[d], laps_to_load, starting_lap, num_laps_predict, d, year, race)
            if p is not None and not p.empty:
                all_preds[d] = p

    if all_preds:
        # Crea grafico combinato
        fig_pace = go.Figure()
        colors = px.colors.qualitative.Light24
        
        for idx, d in enumerate(all_preds.keys()):
            df_p = all_preds[d]
            c = colors[idx % len(colors)]
            fig_pace.add_trace(go.Scatter(
                x=df_p['LapNumber'], 
                y=df_p['Predicted_LapTime_sec'], 
                mode='lines+markers', 
                name=d, 
                line=dict(color=c, width=2)
            ))
            
        fig_pace.update_layout(
            xaxis_title="Numero Giro",
            yaxis_title="Tempo Previsto (secondi)",
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pace, width='stretch')
        
        # Mostra tabelle individuali dentro expander organizzati in colonne
        st.markdown("### 📋 Dettagli Proiezioni Individuali")
        cols = st.columns(4)
        col_idx = 0
        
        for d in available_drivers:
            if d in all_preds:
                with cols[col_idx % 4].expander(f"Dati {d}"):
                    disp = all_preds[d][['LapNumber', 'TyreLife', 'Predicted_LapTime_sec']].copy()
                    disp['Predicted_LapTime'] = disp['Predicted_LapTime_sec'].apply(lambda x: f"{int(x//60)}:{(x%60):06.3f}" if pd.notna(x) else "N/A")
                    st.dataframe(disp[['LapNumber', 'TyreLife', 'Predicted_LapTime']], width='stretch')
                col_idx += 1
    else:
        st.warning("Nessuna previsione disponibile. Controlla i dati o cambia giro di partenza.")

except Exception as e:
    import traceback
    st.error(f"Impossibile completare l'operazione. Dettagli errore: {e}\n\n```\n{traceback.format_exc()}\n```")
