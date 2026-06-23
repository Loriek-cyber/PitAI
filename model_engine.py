import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

def interpolate_telemetry(telemetry: pd.DataFrame, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Allinea le velocità su un asse di distanza comune tramite interpolazione spaziale.
    Risolve il problema del campionamento a frequenze diverse per i sensori.
    """
    # Pulizia dei dati
    if 'Distance' not in telemetry.columns or 'Speed' not in telemetry.columns or telemetry.empty:
        return np.linspace(0, 1000, num_points), np.zeros(num_points)
        
    tel_clean = telemetry.dropna(subset=['Distance', 'Speed'])
        
    min_dist = tel_clean['Distance'].min()
    max_dist = tel_clean['Distance'].max()
    
    # Creazione asse distanza comune
    common_distance = np.linspace(min_dist, max_dist, num_points)
    
    # Interpolazione spaziale della velocità
    speed_interp = np.interp(common_distance, tel_clean['Distance'], tel_clean['Speed'])
    
    return common_distance, speed_interp

def prepare_features(laps: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Prepara il dataset per l'addestramento XGBoost estraendo feature derivate 
    quali l'età della gomma, la mescola, ecc.
    """
    df = laps.copy()
    
    # Calcolo LapTime in secondi (sicuro contro colonne inferite erroneamente come datetime da Pandas per assenza di dati)
    if pd.api.types.is_timedelta64_dtype(df['LapTime']):
        df['LapTime_sec'] = df['LapTime'].dt.total_seconds()
    else:
        df['LapTime_sec'] = np.nan
    
    # Feature Autoregressiva
    df['Prev_LapTime_sec'] = df['LapTime_sec'].shift(1)
    df['Prev_LapTime_sec'] = df['Prev_LapTime_sec'].bfill() # Usa il giro 2 per il giro 1 (o qualifica se pre-iniettato)
    
    # Pulizia fallback per feature deboli (Compound nulli da OpenF1 API)
    df['Compound'] = df['Compound'].fillna('UNKNOWN')
    
    # Gestione fallback per feature telemetriche
    for col in ['AvgThrottle', 'AvgBrake', 'IsSC']:
        if col not in df.columns:
            df[col] = 0.0
            
    df['AvgThrottle'] = df['AvgThrottle'].fillna(0.0)
    df['AvgBrake'] = df['AvgBrake'].fillna(0.0)
    df['IsSC'] = df['IsSC'].fillna(0)
    
    # Rimuoviamo righe con valori nulli critici
    df = df.dropna(subset=['LapTime_sec', 'TyreLife', 'Compound', 'Prev_LapTime_sec'])
    
    # Encoding categorico per la mescola (Compound)
    le = LabelEncoder()
    df['Compound_encoded'] = le.fit_transform(df['Compound'].astype(str))
    
    return df, le

def train_pace_model(df: pd.DataFrame) -> xgb.XGBRegressor:
    """
    Addestra il modello XGBoost per prevedere il tempo sul giro futuro basato su degrado e mescola.
    """
    # Utilizziamo feature note all'inizio del giro per prevedere il tempo finale.
    # Abbiamo inserito n_estimators=200 e learning_rate=0.05 come richiesto.
    features = ['LapNumber', 'TyreLife', 'Compound_encoded', 'Stint', 'Prev_LapTime_sec', 'AvgThrottle', 'AvgBrake', 'IsSC']
    X = df[features]
    y = df['LapTime_sec']
    
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    return model

def predict_future_pace(model: xgb.XGBRegressor, 
                        current_lap: int, 
                        current_tyre_life: int, 
                        current_compound_enc: int, 
                        current_stint: int, 
                        current_laptime: float,
                        avg_throttle: float,
                        avg_brake: float,
                        num_laps: int = 5) -> pd.DataFrame:
    """
    Genera le previsioni per i prossimi N giri simulando l'aumento dell'età della gomma
    in modalità autoregressiva (il tempo previsto del giro N diventa Prev_LapTime del giro N+1).
    """
    future_data = []
    prev_lt = current_laptime
    
    for i in range(1, num_laps + 1):
        feature_cols = ['LapNumber', 'TyreLife', 'Compound_encoded', 'Stint', 'Prev_LapTime_sec', 'AvgThrottle', 'AvgBrake', 'IsSC']
        
        # Costruiamo il df con le feature del singolo giro da prevedere
        features_df = pd.DataFrame([{
            'LapNumber': current_lap + i,
            'TyreLife': current_tyre_life + i,
            'Compound_encoded': current_compound_enc,
            'Stint': current_stint,
            'Prev_LapTime_sec': prev_lt,
            'AvgThrottle': avg_throttle,
            'AvgBrake': avg_brake,
            'IsSC': 0
        }])
        
        # Previsione tempi sul giro (in secondi)
        pred = model.predict(features_df[feature_cols])[0]
        
        future_data.append({
            'LapNumber': current_lap + i,
            'TyreLife': current_tyre_life + i,
            'Compound_encoded': current_compound_enc,
            'Stint': current_stint,
            'Predicted_LapTime_sec': pred
        })
        
        # Autoregressione: la predizione diventa il tempo precedente del prossimo giro
        prev_lt = pred
        
    return pd.DataFrame(future_data)
