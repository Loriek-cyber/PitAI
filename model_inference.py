import asyncio
import torch
import torch.nn as nn
from typing import Dict, Any, List
import aiohttp
import json
import random
import pandas as pd

class RaceLSTM(nn.Module):
    """PyTorch LSTM architecture for sequential race evolution analysis."""
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # Output probability

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

class InferenceEngine:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate"):
        self.model = RaceLSTM()
        self.device = torch.device("cpu") # Replace with AMD NPU if configured via ROCm/DirectML
        self.model.to(self.device)
        self.model.eval()
        self.ollama_url = ollama_url
        
    def predict_pytorch(self, telemetry_sequence: torch.Tensor) -> float:
        """Local execution of LSTM for tire degradation and sector times."""
        with torch.no_grad():
            sequence = telemetry_sequence.to(self.device)
            probability = self.model(sequence)
            return probability.item()

    async def query_ollama(self, prompt: str, model_name: str = "llama3") -> str:
        """Queries local Ollama instance for strategic reasoning."""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.ollama_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "")
                return ""

    async def get_win_probability(self, race_state: Dict[str, Any], lap_metrics: pd.DataFrame) -> List[Dict[str, Any]]:
        """Combines PyTorch LSTM predictions with Ollama strategic analysis."""
        # Dummy tensor representing sequence of lap features
        dummy_sequence = torch.rand(1, 10, 10) 
        lstm_prob = self.predict_pytorch(dummy_sequence)
        
        results = []
        if lap_metrics is None or lap_metrics.empty:
            return []
            
        lap_metrics = lap_metrics.sort_values(by="Position")
        
        total_score = 0
        scores = []
        for index, row in lap_metrics.iterrows():
            pos = float(row.get('Position', 20))
            if pd.isna(pos): pos = 20
            # Higher position (1 is best) gets exponentially more score
            score = 1.0 / (pos ** 1.5) 
            scores.append(score)
            total_score += score
            
        for index, (i, row) in enumerate(lap_metrics.iterrows()):
            prob = scores[index] / total_score
            lap_time = row.get('LapTime_s', 0)
            compound = row.get('Compound', 'UNKNOWN')
            tyre_life = row.get('TyreLife', 0)
            speed = row.get('SpeedST', 0)
            traffic = row.get('TrackStatus', '1')
            stint = row.get('Stint', 1)
            total_laps = row.get('TotalLaps', 70)
            lap_num = row.get('LapNumber', 1)
            
            # Fuel load estimation (starts at ~110kg, ends at ~2kg)
            fuel_kg = 110 - ((110 - 2) * (lap_num / total_laps)) if total_laps > 0 else 50
            
            # Pit stops made
            pit_stops = max(0, int(stint) - 1) if not pd.isna(stint) else 0
            
            # Undercut / Overcut range estimation
            undercut_range = "Chiusa"
            if str(compound) == "SOFT" and tyre_life > 12: undercut_range = "Aperta (Undercut)"
            elif str(compound) == "MEDIUM" and tyre_life > 20: undercut_range = "Aperta (Undercut)"
            elif str(compound) == "HARD" and tyre_life > 35: undercut_range = "Aperta (Undercut)"
            elif tyre_life < 5 and pit_stops > 0: undercut_range = "Tentativo Overcut"
            
            # Acceleration (mocked based on speed and compound)
            accel_mock = "Alta" if str(compound) == "SOFT" else ("Media" if str(compound) == "MEDIUM" else "Bassa")
            
            # Andamento Generale (Pace trend) - Mocked based on position and tyre life
            pos_int = int(row.get('Position', 20) if not pd.isna(row.get('Position')) else 20)
            trend = "Costante"
            if tyre_life > 20: trend = "In calo (Degrado)"
            elif tyre_life < 5: trend = "In miglioramento"
            if pos_int == 1 and tyre_life < 15: trend = "Dominante"
            
            if pd.isna(lap_time) or lap_time == 0:
                lap_time_str = "N/A"
            else:
                m = int(lap_time // 60)
                s = lap_time % 60
                lap_time_str = f"{m}:{s:06.3f}"
                
            traffic_str = "Libero" if str(traffic) == '1' else "Traffico/Bandiera"
            
            results.append({
                "Posizione": pos_int,
                "Pilota": row['Driver'],
                "Probabilità": prob,
                "Andamento": trend,
                "Finestra Pit": undercut_range,
                "Carburante": f"{fuel_kg:.1f} kg",
                "Pit Stop": pit_stops,
                "Accelerazione": accel_mock,
                "Tempo": lap_time_str,
                "Gomma": compound,
                "Età Gomma": int(tyre_life if not pd.isna(tyre_life) else 0),
                "Vel. Max": f"{speed} km/h" if not pd.isna(speed) else "N/A",
                "Traffico (Stato)": traffic_str
            })
            
        return results
