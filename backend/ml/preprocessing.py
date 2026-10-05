import os
import numpy as np
import pandas as pd

def generate_student_dataset(n_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic student academic dataset with realistic collinearity,
    diminishing returns, and realistic exam conditions.
    """
    np.random.seed(random_seed)

    # Latent academic dedication/aptitude factor
    latent = np.random.normal(0, 1.0, n_samples)

    # Core features
    study_hours = np.clip(
        np.round(np.random.normal(4.5 + 0.9 * latent, 1.8, n_samples), 1),
        0.5, 14.0
    )
    attendance = np.clip(
        np.round(np.random.normal(82.0 + 6.0 * latent, 9.5, n_samples), 1),
        40.0, 100.0
    )
    assignment_score = np.clip(
        np.round(np.random.normal(75.0 + 7.5 * latent + 0.8 * study_hours, 9.0, n_samples), 1),
        20.0, 100.0
    )
    previous_marks = np.clip(
        np.round(np.random.normal(70.0 + 10.0 * latent, 11.0, n_samples), 1),
        20.0, 100.0
    )
    mock_test_score = np.clip(
        np.round(np.random.normal(68.0 + 8.5 * latent + 0.15 * previous_marks, 10.0, n_samples), 1),
        20.0, 100.0
    )
    sleep_hours = np.clip(
        np.round(np.random.normal(7.0 - 0.2 * study_hours, 1.1, n_samples), 1),
        4.0, 10.0
    )
    class_participation = np.clip(
        np.round(np.random.normal(6.5 + 0.8 * latent + 0.03 * attendance, 1.5, n_samples), 1),
        1.0, 10.0
    )

    # Realistic relationship with diminishing returns on study hours > 8
    effective_study = np.where(study_hours > 8.0, 8.0 + (study_hours - 8.0) * 0.4, study_hours)

    raw_exam_marks = (
        0.30 * previous_marks +
        0.22 * mock_test_score +
        0.18 * assignment_score +
        1.60 * effective_study +
        0.12 * attendance +
        0.40 * class_participation +
        np.random.normal(0, 3.2, n_samples) # Natural exam-day variance
    )

    exam_marks = np.clip(np.round(raw_exam_marks, 1), 15.0, 100.0)

    df = pd.DataFrame({
        "student_id": [f"STU{1000 + i}" for i in range(n_samples)],
        "study_hours": study_hours,
        "attendance": attendance,
        "assignment_score": assignment_score,
        "previous_marks": previous_marks,
        "mock_test_score": mock_test_score,
        "sleep_hours": sleep_hours,
        "class_participation": class_participation,
        "exam_marks": exam_marks
    })

    return df

def clean_and_preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw student dataset: removes duplicates, imputes missing values,
    and clips numeric features to valid bounds.
    """
    cleaned = df.copy()

    # Drop duplicates if any
    cleaned = cleaned.drop_duplicates()

    # Impute missing values with median
    numeric_cols = [
        "study_hours", "attendance", "assignment_score",
        "previous_marks", "mock_test_score", "sleep_hours",
        "class_participation", "exam_marks"
    ]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            if cleaned[col].isnull().any():
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # Range validations
    if "study_hours" in cleaned.columns:
        cleaned["study_hours"] = cleaned["study_hours"].clip(0.5, 16.0)
    if "attendance" in cleaned.columns:
        cleaned["attendance"] = cleaned["attendance"].clip(0.0, 100.0)
    if "assignment_score" in cleaned.columns:
        cleaned["assignment_score"] = cleaned["assignment_score"].clip(0.0, 100.0)
    if "previous_marks" in cleaned.columns:
        cleaned["previous_marks"] = cleaned["previous_marks"].clip(0.0, 100.0)
    if "mock_test_score" in cleaned.columns:
        cleaned["mock_test_score"] = cleaned["mock_test_score"].clip(0.0, 100.0)
    if "exam_marks" in cleaned.columns:
        cleaned["exam_marks"] = cleaned["exam_marks"].clip(0.0, 100.0)

    return cleaned

if __name__ == "__main__":
    os.makedirs("backend/data", exist_ok=True)
    df = generate_student_dataset(1200)
    df.to_csv("backend/data/students.csv", index=False)
    print(f"Generated {len(df)} student records saved to backend/data/students.csv")
