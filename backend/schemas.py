from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PredictionInput(BaseModel):
    study_hours: float = Field(..., ge=0.5, le=16.0, description="Daily study hours")
    attendance: float = Field(..., ge=0.0, le=100.0, description="Attendance percentage")
    assignment_score: float = Field(..., ge=0.0, le=100.0, description="Assignment score out of 100")
    previous_marks: float = Field(..., ge=0.0, le=100.0, description="Previous exam marks out of 100")
    mock_test_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Optional mock test score")

class PredictionRange(BaseModel):
    min: float
    max: float

class PredictionResponse(BaseModel):
    predicted_marks: float
    prediction_range: PredictionRange
    performance_level: str
    grade: str
    factors: List[Dict[str, Any]]

class WhatIfInput(BaseModel):
    baseline: PredictionInput
    modified: PredictionInput

class WhatIfResponse(BaseModel):
    baseline_predicted_marks: float
    modified_predicted_marks: float
    difference: float
    message: str

class ModelInfoResponse(BaseModel):
    best_model_name: str
    feature_columns: List[str]
    train_size: int
    test_size: int
    trained_at: str
    r2_score: float
    rmse: float
    mae: float

class LeaderboardItem(BaseModel):
    model_name: str
    mae: float
    mse: float
    rmse: float
    r2: float
    is_best: bool
