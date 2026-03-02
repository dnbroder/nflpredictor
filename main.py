"""
NFL Predictor — FastAPI application.
Mirrors GolfPredictor main.py structure.
"""

import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nfl_predictor.feature_engineering import TARGET_COLUMNS
from nfl_predictor.model import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s")
logger = logging.getLogger("nflpredictor")

app = FastAPI(title="NFL Predictor API", version="0.1.0",
              description="ASBM-based NFL prediction system")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Global model cache
_model = None
_features = None
_target = None


def _get_model():
    global _model, _features, _target
    if _model is None:
        _model, _features, _target = load_model()
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    return _model, _features, _target


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    target: Optional[str]
    features_count: Optional[int]
    timestamp: str


class PredictionRequest(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    closing_total: Optional[float] = None
    closing_spread: Optional[float] = None
    features: Optional[dict] = None


class PredictionResponse(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    target: str
    prediction: float
    confidence: float
    bet_recommendation: str
    timestamp: str


@app.get("/health", response_model=HealthResponse)
def health():
    model_loaded = _model is not None
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        target=_target,
        features_count=len(_features) if _features else None,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    model, features, target = _get_model()

    if req.features:
        X = pd.DataFrame([req.features])
        missing = [f for f in features if f not in X.columns]
        for col in missing:
            X[col] = np.nan
        X = X[features]
    else:
        raise HTTPException(status_code=400, detail="Must provide 'features' dict")

    proba = model.predict_proba(X)[:, 1][0]
    confidence = max(proba, 1 - proba)

    if confidence < 0.5238:
        recommendation = "NO BET (below break-even)"
    elif confidence < 0.55:
        recommendation = "LEAN (marginal edge)"
    elif confidence < 0.60:
        recommendation = "BET (moderate edge)"
    else:
        recommendation = "STRONG BET (high confidence)"

    return PredictionResponse(
        game_id=req.game_id,
        home_team=req.home_team,
        away_team=req.away_team,
        target=target,
        prediction=round(float(proba), 4),
        confidence=round(float(confidence), 4),
        bet_recommendation=recommendation,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/markets")
def markets():
    return {"available_markets": list(TARGET_COLUMNS.keys()),
            "active_target": _target or "none (model not loaded)"}


@app.on_event("startup")
async def startup():
    global _model, _features, _target
    try:
        _model, _features, _target = load_model()
        if _model:
            logger.info(f"[API] Model loaded: target={_target}, features={len(_features)}")
        else:
            logger.warning("[API] No model found at startup. Train and deploy a model.")
    except Exception as e:
        logger.warning(f"[API] Model load failed at startup: {e}")