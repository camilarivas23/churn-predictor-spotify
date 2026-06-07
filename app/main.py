"""
FastAPI Application for Spotify Listener Churn Prediction
Provides /health and /predict endpoints
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model loader
from model import load_model


# ===== Pydantic Models =====
class PredictRequest(BaseModel):
    """Input schema for /predict endpoint."""
    days_since_last_listen: float = Field(..., description="Days since most recent listen")
    listening_events_last_7d: int = Field(..., description="Events in last 7 days")
    avg_days_between_listens: float = Field(..., description="Avg days between listens")
    is_active_last_7d: int = Field(..., description="Binary: active in last 7 days")
    total_artist_listens: int = Field(..., description="Total listening events")
    unique_artists_90d: int = Field(..., description="Unique artists in 90d")
    listen_to_artist_ratio: float = Field(..., description="Top artist concentration")
    repeat_listen_rate: float = Field(..., description="Repeat listen rate")
    artist_popularity_avg: float = Field(..., description="Avg artist popularity")
    track_energy_avg: float = Field(..., description="Avg track energy")
    track_danceability_avg: float = Field(..., description="Avg track danceability")
    is_high_engagement_user: int = Field(..., description="Binary: high engagement")
    account_age_days: float = Field(..., description="Account age in days")
    avg_track_tempo: float = Field(..., description="Avg track tempo")


class TopFeature(BaseModel):
    """Top contributing feature."""
    name: str
    importance: float
    value: float


class PredictResponse(BaseModel):
    """Output schema for /predict endpoint."""
    prediction: str = Field(..., description="CHURNED or ACTIVE")
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    confidence: float = Field(..., description="Confidence of prediction")
    top_features: List[TopFeature]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool


# ===== Initialize FastAPI App =====
app = FastAPI(
    title="Spotify Listener Churn Predictor",
    description="Predicts whether a Spotify listener will churn based on listening behavior",
    version="1.0.0"
)

# Global model instance
predictor = None
model_initialized = False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor, model_initialized
    
    try:
        model_path = os.getenv("MODEL_PATH", "./model.pkl")
        features_path = os.getenv("FEATURES_PATH", "./features.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features metadata not found at {features_path}")
        
        predictor = load_model(model_path, features_path)
        model_initialized = True
        logger.info("✓ Model loaded successfully on startup")
        
    except Exception as e:
        logger.error(f"✗ Error loading model on startup: {e}")
        model_initialized = False


# ===== Health Check Endpoint =====
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="OK",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_initialized
    )


# ===== Prediction Endpoint =====
@app.post("/predict", response_model=PredictResponse)
async def predict_churn(request: PredictRequest):
    """Predict listener churn probability.
    
    Takes a listener's feature values and returns churn prediction.
    """
    if not model_initialized:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to feature dictionary
        features_dict = {
            "days_since_last_listen": request.days_since_last_listen,
            "listening_events_last_7d": request.listening_events_last_7d,
            "avg_days_between_listens": request.avg_days_between_listens,
            "is_active_last_7d": request.is_active_last_7d,
            "total_artist_listens": request.total_artist_listens,
            "unique_artists_90d": request.unique_artists_90d,
            "listen_to_artist_ratio": request.listen_to_artist_ratio,
            "repeat_listen_rate": request.repeat_listen_rate,
            "artist_popularity_avg": request.artist_popularity_avg,
            "track_energy_avg": request.track_energy_avg,
            "track_danceability_avg": request.track_danceability_avg,
            "is_high_engagement_user": request.is_high_engagement_user,
            "account_age_days": request.account_age_days,
            "avg_track_tempo": request.avg_track_tempo
        }
        
        # Get prediction
        result = predictor.predict(features_dict)
        
        # Parse top features
        top_features = [
            TopFeature(
                name=f["name"],
                importance=f["importance"],
                value=f["value"]
            )
            for f in result["top_features"]
        ]
        
        response = PredictResponse(
            prediction=result["prediction"],
            churn_probability=result["churn_probability"],
            confidence=result["confidence"],
            top_features=top_features,
            timestamp=datetime.now().isoformat()
        )
        
        # Log prediction
        logger.info(f"Prediction made: {response.prediction} (prob={response.churn_probability:.3f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# ===== Model Info Endpoint =====
@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if not model_initialized:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return predictor.get_model_info()


# ===== Root Endpoint =====
@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Spotify Listener Churn Predictor API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "model-info": "/model-info",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)
