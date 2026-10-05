import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.schemas import (
    PredictionInput,
    PredictionResponse,
    WhatIfInput,
    WhatIfResponse,
    ModelInfoResponse
)
from backend.predictor import predictor
from backend.database.database import get_recent_predictions, clear_predictions

app = FastAPI(
    title="Exam Marks Prediction API",
    description="End-to-End Machine Learning Regression API for Academic Forecasting",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.post("/api/predict", response_model=PredictionResponse)
def predict_exam_marks(data: PredictionInput):
    """Predicts expected exam marks with 95% confidence interval and performance rating."""
    try:
        return predictor.predict(data, persist=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/what-if", response_model=WhatIfResponse)
def what_if_analysis(data: WhatIfInput):
    """Simulates how changes in study time or attendance impact projected score."""
    try:
        return predictor.what_if(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-if simulation error: {str(e)}")

@app.get("/api/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """Returns metadata about the active production model."""
    art = predictor.artifact
    return ModelInfoResponse(
        best_model_name=art["best_model_name"],
        feature_columns=art["feature_columns"],
        train_size=art["train_size"],
        test_size=art["test_size"],
        trained_at=art["trained_at"],
        r2_score=art["r2"],
        rmse=art["rmse"],
        mae=art["mae"]
    )

@app.get("/api/model-metrics")
def get_model_metrics():
    """Returns comparative metrics leaderboard across all evaluated regression models."""
    art = predictor.artifact
    return {
        "leaderboard": art["leaderboard"],
        "all_metrics": art["all_metrics"],
        "test_actual_vs_pred": art["test_actual_vs_pred"],
        "best_model_name": art["best_model_name"]
    }

@app.get("/api/feature-importance")
def get_feature_importance():
    """Returns normalized relative factor weights derived from the trained model."""
    art = predictor.artifact
    return {
        "features": art["feature_importances"],
        "method": art["importance_method"]
    }

@app.get("/api/statistics")
def get_dataset_statistics():
    """Returns dataset correlations and descriptive summary stats."""
    art = predictor.artifact
    return {
        "correlations": art["correlations"],
        "summary_statistics": art["summary_statistics"]
    }

@app.get("/api/predictions")
def list_prediction_history(limit: int = 50):
    """Retrieves recent prediction records stored in SQLite."""
    return get_recent_predictions(limit=limit)

@app.delete("/api/predictions")
def reset_prediction_history():
    """Clears SQLite prediction history."""
    count = clear_predictions()
    return {"status": "success", "cleared_count": count}

# -------------------------------------------------------------
# Frontend Static Files Mount
# -------------------------------------------------------------
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_frontend_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Frontend index.html not found"}
