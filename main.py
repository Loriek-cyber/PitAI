from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any
from data_pipeline import DataPipeline
from model_inference import InferenceEngine

app = FastAPI(title="F1 AI Predictor API", version="1.0.0")

pipeline = DataPipeline()
inference = InferenceEngine()

class RaceContext(BaseModel):
    year: int
    race: str
    lap: int
    is_live: bool = False

@app.on_event("startup")
async def startup_event():
    # Initialize cache or background connections
    pass

@app.get("/api/v1/schedule/{year}")
async def get_schedule(year: int) -> Dict[str, Any]:
    """Endpoint to get all races for a given year."""
    try:
        races = pipeline.get_schedule(year)
        return {"year": year, "races": races}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/laps/{year}/{race}")
async def get_total_laps(year: int, race: str) -> Dict[str, Any]:
    """Endpoint to get total laps for a given race."""
    try:
        laps = await pipeline.get_total_laps(year, race)
        return {"year": year, "race": race, "total_laps": laps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/drivers/{year}/{race}")
async def get_drivers(year: int, race: str) -> Dict[str, Any]:
    """Endpoint to get all drivers for a given race."""
    try:
        drivers = await pipeline.get_drivers(year, race)
        return {"year": year, "race": race, "drivers": drivers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/probability")
async def get_probability(context: RaceContext) -> Dict[str, Any]:
    """Endpoint to query general win probabilities at exact race moments."""
    try:
        race_state = {"year": context.year, "race": context.race, "lap": context.lap}
        lap_metrics_df = await pipeline.get_lap_metrics(context.year, context.race, context.lap)
        
        probs_and_metrics = await inference.get_win_probability(race_state, lap_metrics_df)
        
        return {
            "status": "success",
            "lap": context.lap,
            "data": probs_and_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/head-to-head")
async def get_head_to_head(
    driver1: str = Query(..., min_length=3, max_length=3),
    driver2: str = Query(..., min_length=3, max_length=3),
    year: int = Query(2024),
    race: str = Query(...)
) -> Dict[str, Any]:
    """Endpoint for head-to-head success rates and lap deltas between two drivers."""
    try:
        comparison_df = await pipeline.fetch_driver_head_to_head(driver1, driver2, year, race)
        
        # Calculate a simple success rate based on who has faster laps
        d1_faster = (comparison_df['Delta'] < 0).sum()
        total_laps = len(comparison_df.dropna())
        
        d1_success_rate = (d1_faster / total_laps) if total_laps > 0 else 0
        
        return {
            "driver1": driver1,
            "driver2": driver2,
            "driver1_success_rate": float(d1_success_rate),
            "driver2_success_rate": float(1 - d1_success_rate),
            "lap_data": comparison_df.fillna(0).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
