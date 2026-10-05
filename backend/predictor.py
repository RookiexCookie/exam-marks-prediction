import os
import joblib
import numpy as np
import pandas as pd

from backend.schemas import PredictionInput, PredictionResponse, PredictionRange, WhatIfInput, WhatIfResponse
from backend.database.database import save_prediction

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml", "model.joblib")

class PredictorService:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            from backend.ml.train import train_and_export_models
            self.artifact = train_and_export_models()
        else:
            self.artifact = joblib.load(MODEL_PATH)

        self.pipeline = self.artifact["best_pipeline"]
        self.rmse = self.artifact["rmse"]
        self.feature_columns = self.artifact["feature_columns"]
        self.feature_importances = self.artifact["feature_importances"]

    def _prepare_vector(self, data: PredictionInput) -> pd.DataFrame:
        mock_val = data.mock_test_score
        if mock_val is None:
            # Impute sensibly from previous performance
            mock_val = round(0.55 * data.previous_marks + 0.45 * data.assignment_score, 1)

        row = {
            "study_hours": float(data.study_hours),
            "attendance": float(data.attendance),
            "assignment_score": float(data.assignment_score),
            "previous_marks": float(data.previous_marks),
            "mock_test_score": float(mock_val)
        }
        return pd.DataFrame([row])[self.feature_columns]

    def _get_performance_level(self, mark: float) -> tuple[str, str]:
        if mark >= 85:
            return "Outstanding", "A+"
        elif mark >= 75:
            return "Very Good", "A"
        elif mark >= 65:
            return "Good", "B+"
        elif mark >= 50:
            return "Average", "B"
        elif mark >= 40:
            return "Pass", "C"
        else:
            return "Needs Remedial Action", "F"

    def predict(self, input_data: PredictionInput, persist: bool = True) -> PredictionResponse:
        input_df = self._prepare_vector(input_data)
        raw_pred = self.pipeline.predict(input_df)[0]
        pred_mark = round(float(np.clip(raw_pred, 0.0, 100.0)), 1)

        # 95% Prediction Interval: pred +/- 1.96 * RMSE
        margin = 1.96 * self.rmse
        min_range = round(float(np.clip(pred_mark - margin, 0.0, 100.0)), 1)
        max_range = round(float(np.clip(pred_mark + margin, 0.0, 100.0)), 1)

        perf, grade = self._get_performance_level(pred_mark)

        res = PredictionResponse(
            predicted_marks=pred_mark,
            prediction_range=PredictionRange(min=min_range, max=max_range),
            performance_level=perf,
            grade=grade,
            factors=self.feature_importances
        )

        if persist:
            try:
                save_prediction({
                    "study_hours": input_data.study_hours,
                    "attendance": input_data.attendance,
                    "assignment_score": input_data.assignment_score,
                    "previous_marks": input_data.previous_marks,
                    "mock_test_score": input_data.mock_test_score if input_data.mock_test_score is not None else -1,
                    "predicted_marks": pred_mark,
                    "prediction_range": {"min": min_range, "max": max_range},
                    "performance_level": perf
                })
            except Exception as e:
                print(f"Warning: Failed to save to database: {e}")

        return res

    def what_if(self, whatif: WhatIfInput) -> WhatIfResponse:
        base_res = self.predict(whatif.baseline, persist=False)
        mod_res = self.predict(whatif.modified, persist=False)

        diff = round(mod_res.predicted_marks - base_res.predicted_marks, 1)
        if diff > 0:
            msg = f"Projected increase of +{diff} marks with improved study habits."
        elif diff < 0:
            msg = f"Projected decrease of {diff} marks if dedication drops."
        else:
            msg = "No significant score variance predicted for this change."

        return WhatIfResponse(
            baseline_predicted_marks=base_res.predicted_marks,
            modified_predicted_marks=mod_res.predicted_marks,
            difference=diff,
            message=msg
        )

predictor = PredictorService()
