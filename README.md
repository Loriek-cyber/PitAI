# F1 AI Race Intelligence – Merged Project

Progetto unificato per l'analisi di dati F1 con predizione basata su XGBoost e LSTM.

Combina:
- **OpenF1 API**: dati live e storici con stints, car_data, track_status
- **FastF1**: telemetria ad alta risoluzione, risultati qualifiche, head-to-head

## Struttura

- `data_core.py` – Layer dati unificato (OpenF1 + FastF1)
- `models.py` – Feature engineering, XGBoost, LSTM, probabilità
- `app.py` – Dashboard Streamlit interattiva
- `run_e2e.py` – Test end-to-end da terminale

## Esecuzione

```bash
# Dashboard
streamlit run app.py

# Test E2E
python run_e2e.py --year 2024 --race "Monza" --driver VER

# Unit tests
uv run pytest tests/ -v
```
