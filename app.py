import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="F1 AI Race Intelligence", layout="wide", page_icon="🏎️", initial_sidebar_state="expanded")

# ─── PREMIUM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 40%, #1a1f2e 70%, #0D1117 100%);
        font-family: 'Inter', sans-serif;
        color: #e6edf3;
    }

    /* Sidebar glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.85) !important;
        backdrop-filter: blur(20px) saturate(180%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00D2BE !important;
        font-weight: 700;
    }

    /* Headers with glow */
    h1 {
        background: linear-gradient(90deg, #E10600, #FF8700, #FFF200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 2.4rem !important;
        letter-spacing: -0.5px;
        text-shadow: none;
    }
    h2, h3 {
        color: #58a6ff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(88,166,255,0.3);
    }

    /* Metric cards glassmorphism */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 210, 190, 0.3);
        box-shadow: 0 8px 32px rgba(0, 210, 190, 0.15);
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.7rem !important;
        letter-spacing: 1.2px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-weight: 800;
        font-size: 1.8rem !important;
    }

    /* DataFrame hover */
    div[data-testid="stDataFrame"] {
        transition: all 0.3s ease;
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stDataFrame"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 210, 190, 0.12);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8b949e;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f0f6fc;
        background: rgba(255,255,255,0.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(225,6,0,0.15), rgba(255,135,0,0.15)) !important;
        color: #FF8700 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #E10600 0%, #FF8700 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(225, 6, 0, 0.45) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* Slider */
    .stSlider > div > div { color: #00D2BE; }
</style>
""", unsafe_allow_html=True)

# ─── INIT ──────────────────────────────────────────────────────────────────────
from data_core import DataCore
from models import ModelEngine

@st.cache_resource
def init_systems():
    return DataCore(), ModelEngine()

data_core, model_engine = init_systems()

def fmt_laptime(seconds):
    """Format seconds to M:SS.mmm"""
    if pd.isna(seconds) or seconds == 0:
        return "N/A"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:06.3f}"

F1_COLORS = ['#E10600', '#00D2BE', '#FF8700', '#FFF200', '#0090FF',
             '#6CD3BF', '#F596C8', '#B6BABD', '#2B4562', '#F58020',
             '#9B0000', '#005AFF', '#006F62', '#900000', '#FFFFFF',
             '#FF6700', '#469BFF', '#C92D4B', '#2293D1', '#1E6CA1']

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏎️ Controllo Missione")
year = st.sidebar.selectbox("🗓️ Stagione", list(range(2026, 2019, -1)), index=1)

@st.cache_data(show_spinner=False)
def load_schedule(y):
    try:
        schedule = data_core.get_schedule(y)
        if isinstance(schedule, pd.DataFrame) and "EventName" in schedule.columns:
            return schedule["EventName"].tolist()
        return list(schedule)
    except:
        return ["Monza", "Silverstone", "Spa"]

available_races = load_schedule(year)
race = st.sidebar.selectbox("🏁 Gran Premio", available_races)

is_live = st.sidebar.checkbox("🔴 Modalità Live", value=False)

@st.cache_data(show_spinner=False)
def load_drivers(y, r):
    try: return data_core.get_drivers(y, r)
    except: return ["VER", "HAM", "NOR", "LEC", "SAI"]

available_drivers = load_drivers(year, race)
total_laps = data_core.get_total_laps(year, race)

st.sidebar.divider()
st.sidebar.markdown("### 👥 Selezione Piloti")
sel_drivers = st.sidebar.multiselect(
    "Piloti per Analisi & Pace",
    available_drivers,
    default=available_drivers[:3] if len(available_drivers) >= 3 else available_drivers
)

st.sidebar.divider()
st.sidebar.markdown("### ⚔️ Head-to-Head")
d1 = st.sidebar.selectbox("Pilota 1", available_drivers, index=0)
d2 = st.sidebar.selectbox("Pilota 2", available_drivers, index=min(1, len(available_drivers)-1))

st.sidebar.divider()
st.sidebar.caption("F1 AI Race Intelligence • FIA Project 2025")

# ─── HELPER: FETCH WITH CACHE ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_laps(y, r, drv, live):
    return data_core.get_laps(y, r, drv, is_live=live)

@st.cache_data(show_spinner=False)
def fetch_telemetry(y, r, drv, live):
    _fastest, tel = data_core.get_telemetry(y, r, drv, is_live=live)
    return tel

# ─── TITLE ─────────────────────────────────────────────────────────────────────
st.title("F1 AI Race Intelligence")
st.markdown(f"Stagione **{year}** • Gran Premio di **{race}** • {'🔴 LIVE' if is_live else '📂 Storico'}")

# ─── 5 TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏁 Overview", "🔮 Predizione Pace", "⚡ Speed Trace", "⚔️ Head-to-Head", "📊 Analisi Giro"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW & PROBABILITÀ
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giri Totali", total_laps)
    c2.metric("Piloti", len(available_drivers))
    c3.metric("Stagione", year)
    c4.metric("Gran Premio", str(race)[:20])

    st.divider()
    st.subheader("🎯 Probabilità di Vittoria per Giro")

    sel_lap = st.slider("Seleziona il giro da analizzare", 1, max(total_laps, 1), min(10, total_laps))

    if st.button("Calcola Probabilità", key="btn_prob"):
        with st.spinner("Analisi in corso..."):
            try:
                metrics_df = data_core.get_lap_metrics(year, race, sel_lap)
                if metrics_df is not None and not metrics_df.empty:
                    stats = model_engine.get_real_win_probability(metrics_df, {})
                    if stats:
                        df_stats = pd.DataFrame(stats).sort_values("Probabilità", ascending=True)
                        # Format probability as percentage for display
                        df_stats["Prob_display"] = df_stats["Probabilità"].apply(lambda x: f"{x:.1%}")

                        fig = px.bar(df_stats, x="Probabilità", y="Pilota", orientation='h',
                                     color="Probabilità", color_continuous_scale=["#161B22", "#E10600", "#FF8700"],
                                     title=f"Probabilità di Vittoria • Giro {sel_lap}")
                        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                                          paper_bgcolor="rgba(0,0,0,0)", yaxis_categoryorder='total ascending',
                                          coloraxis_showscale=False, height=max(400, len(df_stats)*30))
                        st.plotly_chart(fig, use_container_width=True)

                        # Display the rich table
                        display_cols = [c for c in ["Posizione", "Pilota", "Prob_display", "Tempo", "Gomma",
                                                      "Età Gomma", "Andamento", "Finestra Pit", "Carburante",
                                                      "Pit Stop", "Vel. Max", "Traffico (Stato)"]
                                        if c in df_stats.columns]
                        df_show = df_stats[display_cols].sort_values("Posizione")
                        df_show = df_show.rename(columns={"Prob_display": "Probabilità"})
                        st.dataframe(df_show, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Nessuna statistica disponibile per questo giro.")
                else:
                    st.warning("Dati non disponibili per il giro selezionato.")
            except Exception as e:
                st.error(f"Errore: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: PREDIZIONE PASSO GARA (XGBOOST)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔮 Previsione Passo Gara (XGBoost)")
    st.markdown("Addestra un modello individuale per ogni pilota e prevede il passo gara futuro in modalità autoregressiva.")

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    laps_to_load = col_cfg1.number_input("Giri da caricare", 1, total_laps, total_laps, key="ltl")
    starting_lap = col_cfg2.number_input("Giro di partenza (0 = Qualifica)", 0, laps_to_load, 0, key="sl")
    num_predict = col_cfg3.number_input("Giri da prevedere", 1, 100, max(1, total_laps - starting_lap), key="np")

    if st.button("🧠 Addestra & Prevedi", key="btn_pace"):
        if not sel_drivers:
            st.warning("Seleziona almeno un pilota dalla sidebar.")
        else:
            all_preds = {}
            progress = st.progress(0, text="Preparazione modelli...")

            for idx, drv in enumerate(sel_drivers):
                progress.progress((idx) / len(sel_drivers), text=f"Training modello per {drv}...")
                try:
                    df_laps = fetch_laps(year, race, drv, is_live)
                    if df_laps is None or df_laps.empty:
                        continue

                    df_prep, le = model_engine.prepare_pace_features(df_laps)
                    df_train = df_prep[df_prep['LapNumber'] <= laps_to_load]
                    if df_train.empty:
                        continue

                    model = model_engine.train_pace_model(df_train)

                    if starting_lap == 0:
                        try:
                            q_session = data_core.load_session(year, str(race), 'Qualifying')
                            q_lap = data_core.get_qualy_fastest_lap(q_session, drv) if q_session else None
                        except Exception:
                            q_lap = None
                        c_lap, c_tyre, c_stint = 0, 0, 1
                        c_comp = int(df_train.iloc[0]['Compound_encoded']) if not df_train.empty else 0
                        c_lt = q_lap if q_lap else (float(df_train.iloc[0]['LapTime_sec']) if not df_train.empty else 90.0)
                        avg_t = float(df_train['AvgThrottle'].mean()) if 'AvgThrottle' in df_train.columns else 0.0
                        avg_b = float(df_train['AvgBrake'].mean()) if 'AvgBrake' in df_train.columns else 0.0
                    else:
                        subset = df_train[df_train['LapNumber'] <= starting_lap]
                        if not subset.empty:
                            last = subset.iloc[-1]
                            c_lap = int(last['LapNumber'])
                            c_tyre = int(last['TyreLife'])
                            c_comp = int(last['Compound_encoded'])
                            c_stint = int(last['Stint'])
                            c_lt = float(last['LapTime_sec'])
                            avg_t = float(subset['AvgThrottle'].mean())
                            avg_b = float(subset['AvgBrake'].mean())
                        else:
                            continue

                    preds = model_engine.predict_future_pace_from_state(
                        model, c_lap, c_tyre, c_comp, c_stint, c_lt, avg_t, avg_b, num_predict
                    )
                    if preds is not None and not preds.empty:
                        all_preds[drv] = preds
                except Exception as e:
                    st.warning(f"Errore per {drv}: {e}")

            progress.empty()

            if all_preds:
                fig_pace = go.Figure()
                for idx, (drv, df_p) in enumerate(all_preds.items()):
                    c = F1_COLORS[idx % len(F1_COLORS)]
                    fig_pace.add_trace(go.Scatter(
                        x=df_p['LapNumber'], y=df_p['Predicted_LapTime_sec'],
                        mode='lines+markers', name=drv, line=dict(color=c, width=2.5),
                        marker=dict(size=5)
                    ))
                fig_pace.update_layout(
                    template="plotly_dark", title="Previsione Passo Gara",
                    xaxis_title="Numero Giro", yaxis_title="Tempo Previsto (sec)",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=1.08), hovermode="x unified", height=500
                )
                st.plotly_chart(fig_pace, use_container_width=True)

                st.markdown("### 📋 Dettagli per Pilota")
                cols = st.columns(min(4, len(all_preds)))
                for i, (drv, df_p) in enumerate(all_preds.items()):
                    with cols[i % len(cols)].expander(f"📊 {drv}", expanded=False):
                        disp = df_p[['LapNumber', 'TyreLife', 'Predicted_LapTime_sec']].copy()
                        disp['Tempo'] = disp['Predicted_LapTime_sec'].apply(fmt_laptime)
                        st.dataframe(disp[['LapNumber', 'TyreLife', 'Tempo']], use_container_width=True, hide_index=True)
            else:
                st.warning("Nessuna previsione generata. Controlla i piloti selezionati o i dati disponibili.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: SPEED TRACE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("⚡ Speed Trace — Giro Più Veloce")
    st.markdown("Confronto della velocità lungo il tracciato. I dati vengono caricati prioritariamente dalla cache locale FastF1.")

    if st.button("🏎️ Carica Telemetria", key="btn_tel"):
        if not sel_drivers:
            st.warning("Seleziona almeno un pilota dalla sidebar.")
        else:
            fig_speed = go.Figure()
            found = False
            progress = st.progress(0, text="Caricamento telemetria...")

            for idx, drv in enumerate(sel_drivers):
                progress.progress((idx) / len(sel_drivers), text=f"Telemetria {drv}...")
                try:
                    tel = fetch_telemetry(year, race, drv, is_live)
                    if tel is not None and not tel.empty:
                        dist, speed = model_engine.interpolate_telemetry(tel)
                        c = F1_COLORS[idx % len(F1_COLORS)]
                        fig_speed.add_trace(go.Scatter(
                            x=dist, y=speed, mode='lines', name=drv,
                            line=dict(color=c, width=2)
                        ))
                        found = True
                except Exception as e:
                    st.warning(f"Telemetria non disponibile per {drv}: {e}")

            progress.empty()

            if found:
                fig_speed.update_layout(
                    template="plotly_dark", xaxis_title="Distanza (m)", yaxis_title="Velocità (km/h)",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified", height=500,
                    legend=dict(orientation="h", y=1.05, xanchor="right", x=1)
                )
                st.plotly_chart(fig_speed, use_container_width=True)
            else:
                st.warning("Nessun dato telemetrico trovato. Prova un'altra gara o altri piloti.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: HEAD-TO-HEAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"⚔️ Head-to-Head: {d1} vs {d2}")

    if st.button("⚡ Confronta", key="btn_h2h"):
        with st.spinner("Calcolo delta giro per giro..."):
            try:
                h2h_df = data_core.fetch_driver_head_to_head(d1, d2, year, race)
                if h2h_df is not None and not h2h_df.empty:
                    d1_faster = (h2h_df['Delta'] < 0).sum()
                    d2_faster = (h2h_df['Delta'] > 0).sum()
                    avg_delta = h2h_df['Delta'].mean()

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric(f"🏆 {d1} più veloce", f"{d1_faster} giri")
                    mc2.metric(f"🏆 {d2} più veloce", f"{d2_faster} giri")
                    mc3.metric("Delta Medio", f"{avg_delta:+.3f}s")

                    st.divider()

                    # Use 'LapNumber' which is the canonical column name
                    x_col = "Lap" if "Lap" in h2h_df.columns else "LapNumber"

                    fig_delta = px.bar(h2h_df, x=x_col, y="Delta",
                                       color="Delta", color_continuous_scale="RdBu_r",
                                       title=f"Delta per Giro (Negativo = {d1} più veloce)")
                    fig_delta.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                                            paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, height=400,
                                            xaxis_title="Giro")
                    st.plotly_chart(fig_delta, use_container_width=True)

                    h2h_df['CumDelta'] = h2h_df['Delta'].cumsum()
                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=h2h_df[x_col], y=h2h_df['CumDelta'],
                        mode='lines', name='Delta Cumulativo',
                        line=dict(color='#00D2BE', width=3),
                        fill='tozeroy', fillcolor='rgba(0,210,190,0.1)'
                    ))
                    fig_cum.add_hline(y=0, line_dash="dash", line_color="#E10600", opacity=0.5)
                    fig_cum.update_layout(
                        template="plotly_dark", title="Delta Cumulativo",
                        xaxis_title="Giro", yaxis_title="Secondi Cumulativi",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=350
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
                else:
                    st.warning("Dati insufficienti per il confronto H2H.")
            except Exception as e:
                st.error(f"Errore nel confronto: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: ANALISI GIRO DETTAGLIATA
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📊 Analisi Dettagliata per Giro")

    lap_target = st.number_input("Seleziona il giro", 1, max(total_laps, 80), 10, key="lap_detail")

    if st.button("🔍 Analizza Giro", key="btn_detail"):
        with st.spinner("Caricamento dati del giro..."):
            try:
                metrics_df = data_core.get_lap_metrics(year, race, lap_target)
                if metrics_df is not None and not metrics_df.empty:
                    stats = model_engine.get_real_win_probability(metrics_df, {})
                    if stats:
                        df_display = pd.DataFrame(stats)

                        # Format probability for display
                        df_display["Prob_display"] = df_display["Probabilità"].apply(lambda x: f"{x:.1%}")

                        # Show the rich data table
                        display_cols = [c for c in ["Posizione", "Pilota", "Prob_display", "Tempo", "Gomma",
                                                      "Età Gomma", "Andamento", "Finestra Pit", "Carburante",
                                                      "Pit Stop", "Accelerazione", "Vel. Max", "Traffico (Stato)"]
                                        if c in df_display.columns]
                        df_show = df_display[display_cols].copy()
                        df_show = df_show.rename(columns={"Prob_display": "Probabilità"})
                        st.dataframe(df_show, use_container_width=True, hide_index=True)

                        ch1, ch2 = st.columns(2)

                        with ch1:
                            if 'Gomma' in df_display.columns:
                                compound_counts = df_display['Gomma'].value_counts()
                                fig_pie = px.pie(values=compound_counts.values, names=compound_counts.index,
                                                 title="Distribuzione Mescole",
                                                 color_discrete_sequence=['#E10600', '#FFF200', '#FFFFFF', '#FF8700'])
                                fig_pie.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                                                       paper_bgcolor="rgba(0,0,0,0)", height=350)
                                st.plotly_chart(fig_pie, use_container_width=True)

                        with ch2:
                            if 'Posizione' in df_display.columns and 'Pilota' in df_display.columns:
                                df_sorted = df_display.sort_values('Posizione', ascending=True)
                                fig_pos = px.bar(df_sorted, x='Posizione', y='Pilota', orientation='h',
                                                 title="Classifica al Giro", color='Posizione',
                                                 color_continuous_scale=['#00D2BE', '#161B22'])
                                fig_pos.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                                                       paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
                                                       yaxis_categoryorder='total descending', height=350)
                                st.plotly_chart(fig_pos, use_container_width=True)
                    else:
                        st.warning("Nessun dato disponibile per questo giro.")
                else:
                    st.warning("Nessun dato disponibile per questo giro.")
            except Exception as e:
                st.error(f"Errore nel caricamento: {e}")
