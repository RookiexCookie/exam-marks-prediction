import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.data_generator import FEATURE_COLUMNS, TARGET_COLUMN

AVAILABLE_MODELS = {
    "Linear Regression": {
        "class": LinearRegression,
        "params": {},
        "description": "Classic parametric linear model. Fast, highly interpretable with direct feature weights."
    },
    "Random Forest Regressor": {
        "class": RandomForestRegressor,
        "params": {"n_estimators": 120, "max_depth": 8, "random_state": 42},
        "description": "Ensemble of decision trees. Captures non-linear relationships and diminishing returns."
    },
    "Gradient Boosting Regressor": {
        "class": GradientBoostingRegressor,
        "params": {"n_estimators": 100, "learning_rate": 0.08, "max_depth": 4, "random_state": 42},
        "description": "Sequential boosting model focusing iteratively on error residuals for high predictive accuracy."
    },
    "Ridge Regression": {
        "class": Ridge,
        "params": {"alpha": 1.5},
        "description": "Regularized linear regression with L2 penalty, reducing overfitting and multi-collinearity."
    }
}

def get_grade_info(mark: float) -> dict:
    """Returns letter grade, classification, and status badge color for an exam score."""
    if mark >= 90:
        return {"grade": "O (Outstanding)", "class": "First Class with Distinction", "status": "Outstanding", "color": "#10B981"}
    elif mark >= 80:
        return {"grade": "A+ (Excellent)", "class": "First Class with Distinction", "status": "Excellent", "color": "#10B981"}
    elif mark >= 70:
        return {"grade": "A (Very Good)", "class": "First Class", "status": "Very Good", "color": "#3B82F6"}
    elif mark >= 60:
        return {"grade": "B+ (Good)", "class": "First Class", "status": "Good", "color": "#6366F1"}
    elif mark >= 50:
        return {"grade": "B (Above Average)", "class": "Second Class", "status": "Above Average", "color": "#F59E0B"}
    elif mark >= 40:
        return {"grade": "C (Pass)", "class": "Pass Division", "status": "Pass", "color": "#EAB308"}
    else:
        return {"grade": "F (Fail / Critical)", "class": "Needs Improvement", "status": "At Risk", "color": "#EF4444"}

def build_model_pipeline(model_name: str, custom_params: dict | None = None) -> Pipeline:
    """Builds a scikit-learn Pipeline with StandardScaler and selected Regressor."""
    cfg = AVAILABLE_MODELS.get(model_name, AVAILABLE_MODELS["Linear Regression"])
    model_cls = cfg["class"]
    params = cfg["params"].copy()
    if custom_params:
        params.update(custom_params)

    estimator = model_cls(**params)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", estimator)
    ])
    return pipeline

def train_and_eval_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    custom_params: dict | None = None
) -> dict:
    """Trains pipeline, computes MAE, MSE, RMSE, R2, and prediction residuals."""
    pipeline = build_model_pipeline(model_name, custom_params)
    pipeline.fit(X_train, y_train)

    y_train_pred = np.clip(pipeline.predict(X_train), 0.0, 100.0)
    y_test_pred = np.clip(pipeline.predict(X_test), 0.0, 100.0)

    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    residuals = y_test.values - y_test_pred

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "train_r2": float(train_r2),
        "train_rmse": float(train_rmse),
        "y_test": y_test.values,
        "y_test_pred": y_test_pred,
        "residuals": residuals,
        "X_test": X_test
    }

def compare_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[pd.DataFrame, dict]:
    """Trains and compares all available models, returns leaderboard and results map."""
    rows = []
    models_dict = {}

    for name in AVAILABLE_MODELS.keys():
        res = train_and_eval_model(name, X_train, y_train, X_test, y_test)
        models_dict[name] = res
        rows.append({
            "Model": name,
            "R² Score": f"{res['r2']:.4f}",
            "RMSE (Marks)": f"{res['rmse']:.2f}",
            "MAE (Marks)": f"{res['mae']:.2f}",
            "MSE": f"{res['mse']:.2f}",
            "Train R²": f"{res['train_r2']:.4f}",
            "_r2_val": res["r2"],
            "_rmse_val": res["rmse"],
        })

    leaderboard_df = pd.DataFrame(rows).sort_values(by="_r2_val", ascending=False).reset_index(drop=True)
    return leaderboard_df, models_dict

def predict_exam_marks(
    pipeline: Pipeline,
    input_dict: dict,
    rmse: float | None = None
) -> dict:
    """
    Predicts exam mark for a single student input, clips to [0, 100],
    and computes 95% confidence/prediction intervals using RMSE.
    """
    input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
    raw_pred = pipeline.predict(input_df)[0]
    pred_mark = float(np.clip(raw_pred, 0.0, 100.0))

    # Fallback RMSE if not given
    sigma = rmse if (rmse is not None and rmse > 0) else 3.5

    # 95% Prediction Interval: pred +/- 1.96 * sigma
    lower_bound = float(np.clip(pred_mark - 1.96 * sigma, 0.0, 100.0))
    upper_bound = float(np.clip(pred_mark + 1.96 * sigma, 0.0, 100.0))
    # 68% (1-sigma) interval
    sigma_lower = float(np.clip(pred_mark - sigma, 0.0, 100.0))
    sigma_upper = float(np.clip(pred_mark + sigma, 0.0, 100.0))

    grade_info = get_grade_info(pred_mark)

    return {
        "predicted_mark": round(pred_mark, 1),
        "lower_bound_95": round(lower_bound, 1),
        "upper_bound_95": round(upper_bound, 1),
        "lower_bound_68": round(sigma_lower, 1),
        "upper_bound_68": round(sigma_upper, 1),
        "rmse_used": round(sigma, 2),
        "grade_info": grade_info
    }

def get_feature_importance_df(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Extracts normalized feature importance or standardized coefficients."""
    estimator = pipeline.named_steps["regressor"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        method = "Gini / Impurity Reduction"
    elif hasattr(estimator, "coef_"):
        # Absolute standardized coefficients
        raw_coefs = np.abs(estimator.coef_)
        total = np.sum(raw_coefs)
        importances = raw_coefs / (total if total > 0 else 1.0)
        method = "Normalized |Beta| Coefs"
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)
        method = "Uniform"

    df = pd.DataFrame({
        "Feature": [f.replace("_", " ").title() for f in feature_names],
        "Feature_Key": feature_names,
        "Importance": importances,
        "Importance_Pct": (importances * 100.0).round(1)
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    df["Method"] = method
    return df

def generate_student_recommendations(inputs: dict, pred_result: dict) -> list[dict]:
    """Generates personalized actionable recommendations for the student."""
    recs = []
    mark = pred_result["predicted_mark"]
    hours = inputs.get("study_hours", 4.0)
    attendance = inputs.get("attendance_percentage", 80.0)
    internal = inputs.get("internal_marks", 35.0)
    assignments = inputs.get("assignment_score", 75.0)
    study_days = inputs.get("study_days", 30)

    # 1. Study Hours Advice
    if hours < 3.0:
        recs.append({
            "type": "warning",
            "icon": "⚠️",
            "title": "Low Daily Study Time",
            "message": f"Currently studying {hours} hrs/day. Increasing by 1.5 to 2.0 hours daily can potentially boost your score by 6 to 9 marks.",
            "impact": "+6 to 9 Marks"
        })
    elif hours >= 7.0:
        recs.append({
            "type": "success",
            "icon": "🌟",
            "title": "Strong Study Discipline",
            "message": f"Your daily study time of {hours} hrs is commendable! Ensure balanced breaks using the Pomodoro technique to prevent burnout.",
            "impact": "Maintain Consistency"
        })
    else:
        recs.append({
            "type": "info",
            "icon": "💡",
            "title": "Incremental Study Boost",
            "message": f"At {hours} hrs/day, adding just 45 minutes of focused problem solving daily can elevate you to the next grade bracket.",
            "impact": "+3 to 5 Marks"
        })

    # 2. Attendance Advice
    if attendance < 75.0:
        recs.append({
            "type": "danger",
            "icon": "🚨",
            "title": "Critical Attendance Alert (< 75%)",
            "message": f"Your attendance is {attendance}%, which is below the standard university mandate. Attending upcoming lectures is urgent to avoid detention and gain internal marks.",
            "impact": "Crucial Requirement"
        })
    elif attendance >= 85.0:
        recs.append({
            "type": "success",
            "icon": "✅",
            "title": "Excellent Lecture Engagement",
            "message": f"{attendance}% attendance gives you strong continuous exposure to curriculum topics and professor hints for exams.",
            "impact": "Great Asset"
        })

    # 3. Internal Assessment Advice
    if internal < 30.0:
        recs.append({
            "type": "warning",
            "icon": "📝",
            "title": "Internal Assessment Needs Attention",
            "message": f"Scoring {internal}/50 in internals creates a deficit. Review past mistakes with your professor and score higher in upcoming continuous tests.",
            "impact": "+5 to 10 Marks"
        })

    # 4. Preparation Days
    if study_days < 15:
        recs.append({
            "type": "info",
            "icon": "📅",
            "title": "Short Prep Window Ahead",
            "message": f"With only {study_days} study days, switch to high-yield revision: solve 5 past year question papers and focus on 80/20 topics.",
            "impact": "High Yield"
        })

    # 5. Overall Target Milestone
    if mark >= 85:
        recs.append({
            "type": "success",
            "icon": "🎯",
            "title": "Distinction Trajectory",
            "message": "You are projected to score in the top percentile! Maintain your schedule and focus on mock tests to secure a university rank.",
            "impact": "Goal: Rank 1"
        })
    elif mark < 45:
        recs.append({
            "type": "danger",
            "icon": "🆘",
            "title": "Immediate Academic Recovery Required",
            "message": "You are borderline or below passing criteria (40). Combine peer tutoring, daily 4-hour revision, and solved model test papers immediately.",
            "impact": "Immediate Focus"
        })

    return recs

def simulate_target_goal(
    pipeline: Pipeline,
    current_inputs: dict,
    target_mark: float,
    rmse: float = 3.5
) -> dict:
    """
    Simulates how much more study hours or attendance is needed
    to reach the desired target mark.
    """
    current_pred = predict_exam_marks(pipeline, current_inputs, rmse)["predicted_mark"]
    gap = target_mark - current_pred

    if gap <= 0:
        return {
            "status": "achieved",
            "current_pred": current_pred,
            "target_mark": target_mark,
            "gap": 0.0,
            "message": f"Congratulations! Your predicted score ({current_pred:.1f}) already meets or exceeds your goal ({target_mark:.1f})."
        }

    # Simulate required study hours (iterative grid search)
    needed_hours = None
    test_inputs = current_inputs.copy()
    for h in np.arange(current_inputs["study_hours"], 14.1, 0.2):
        test_inputs["study_hours"] = round(float(h), 2)
        p = predict_exam_marks(pipeline, test_inputs, rmse)["predicted_mark"]
        if p >= target_mark:
            needed_hours = round(float(h), 1)
            break

    # Simulate required attendance
    needed_attendance = None
    test_inputs2 = current_inputs.copy()
    for a in np.arange(current_inputs["attendance_percentage"], 100.1, 0.5):
        test_inputs2["attendance_percentage"] = round(float(a), 1)
        p = predict_exam_marks(pipeline, test_inputs2, rmse)["predicted_mark"]
        if p >= target_mark:
            needed_attendance = round(float(a), 1)
            break

    return {
        "status": "in_progress",
        "current_pred": current_pred,
        "target_mark": target_mark,
        "gap": round(gap, 1),
        "needed_hours": needed_hours,
        "current_hours": current_inputs["study_hours"],
        "needed_attendance": needed_attendance,
        "current_attendance": current_inputs["attendance_percentage"]
    }
