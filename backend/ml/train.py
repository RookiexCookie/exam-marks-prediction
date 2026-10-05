import os
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from backend.ml.preprocessing import generate_student_dataset, clean_and_preprocess_dataset
from backend.ml.evaluate import evaluate_predictions

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "students.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")

FEATURE_COLUMNS = [
    "study_hours",
    "attendance",
    "assignment_score",
    "previous_marks",
    "mock_test_score"
]
TARGET_COLUMN = "exam_marks"

def train_and_export_models():
    # 1. Load or Generate Dataset
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"Generating synthetic dataset at {DATA_PATH}...")
        raw_df = generate_student_dataset(n_samples=1200, random_seed=42)
        raw_df.to_csv(DATA_PATH, index=False)
    else:
        raw_df = pd.read_csv(DATA_PATH)

    df = clean_and_preprocess_dataset(raw_df)

    # 2. Features and Target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 3. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 4. Define Candidate Regressors
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    }

    model_results = {}
    best_model_name = None
    best_r2 = -float("inf")

    # 5. Train & Evaluate
    for name, regressor in candidate_models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", regressor)
        ])

        pipeline.fit(X_train, y_train)
        y_test_pred = np.clip(pipeline.predict(X_test), 0.0, 100.0)
        metrics = evaluate_predictions(y_test, y_test_pred)

        model_results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "test_sample_preds": [
                {"actual": float(a), "predicted": round(float(p), 1)}
                for a, p in zip(y_test.values[:60], y_test_pred[:60])
            ]
        }

        # Compare based on R2 score
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_model_name = name

    print(f"Training Complete. Best Model Selected: {best_model_name} (R² = {best_r2})")

    # 6. Extract Feature Importance for Best Model
    best_pipeline = model_results[best_model_name]["pipeline"]
    estimator = best_pipeline.named_steps["regressor"]

    if hasattr(estimator, "feature_importances_"):
        raw_importances = estimator.feature_importances_
        method = "Feature Importances (Tree Gini/Variance Reduction)"
    elif hasattr(estimator, "coef_"):
        raw_importances = np.abs(estimator.coef_)
        method = "Normalized |Standardized Coefficients|"
    else:
        raw_importances = np.ones(len(FEATURE_COLUMNS))
        method = "Uniform"

    norm_importances = raw_importances / np.sum(raw_importances)
    feature_importance_list = [
        {
            "feature": f,
            "label": f.replace("_", " ").title(),
            "importance": round(float(imp), 4),
            "percentage": round(float(imp * 100), 1)
        }
        for f, imp in zip(FEATURE_COLUMNS, norm_importances)
    ]
    feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)

    # 7. Dataset Summary Statistics and Correlations
    corr_matrix = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr().round(3).to_dict()
    summary_stats = df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().round(2).to_dict()

    # Model Leaderboard for UI
    leaderboard = [
        {
            "model_name": name,
            "mae": res["metrics"]["mae"],
            "mse": res["metrics"]["mse"],
            "rmse": res["metrics"]["rmse"],
            "r2": res["metrics"]["r2"],
            "is_best": (name == best_model_name)
        }
        for name, res in model_results.items()
    ]
    leaderboard.sort(key=lambda x: x["r2"], reverse=True)

    # 8. Package Artifact
    artifact = {
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "rmse": model_results[best_model_name]["metrics"]["rmse"],
        "mae": model_results[best_model_name]["metrics"]["mae"],
        "r2": model_results[best_model_name]["metrics"]["r2"],
        "feature_importances": feature_importance_list,
        "importance_method": method,
        "leaderboard": leaderboard,
        "all_metrics": {name: res["metrics"] for name, res in model_results.items()},
        "test_actual_vs_pred": model_results[best_model_name]["test_sample_preds"],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "correlations": corr_matrix,
        "summary_statistics": summary_stats
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Artifact successfully saved to {MODEL_PATH}")
    return artifact

if __name__ == "__main__":
    train_and_export_models()
