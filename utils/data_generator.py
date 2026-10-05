import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "student_data.csv")

FEATURE_COLUMNS = [
    "study_hours",
    "attendance_percentage",
    "assignment_score",
    "previous_exam_marks",
    "internal_marks",
    "study_days",
]
TARGET_COLUMN = "exam_marks"

FEATURE_METADATA = {
    "study_hours": {
        "label": "Daily Study Hours",
        "min": 0.5,
        "max": 14.0,
        "step": 0.5,
        "default": 4.5,
        "unit": "hours/day",
        "description": "Average hours spent studying per day"
    },
    "attendance_percentage": {
        "label": "Attendance Rate",
        "min": 40.0,
        "max": 100.0,
        "step": 1.0,
        "default": 82.0,
        "unit": "%",
        "description": "Percentage of lecture and lab classes attended"
    },
    "assignment_score": {
        "label": "Assignment Score",
        "min": 20.0,
        "max": 100.0,
        "step": 1.0,
        "default": 78.0,
        "unit": "/ 100",
        "description": "Average marks across continuous home assignments"
    },
    "previous_exam_marks": {
        "label": "Previous Exam Marks",
        "min": 20.0,
        "max": 100.0,
        "step": 1.0,
        "default": 72.0,
        "unit": "/ 100",
        "description": "Marks scored in the preceding semester examination"
    },
    "internal_marks": {
        "label": "Internal Assessment Marks",
        "min": 10.0,
        "max": 50.0,
        "step": 1.0,
        "default": 38.0,
        "unit": "/ 50",
        "description": "Mid-term internal assessment score out of 50"
    },
    "study_days": {
        "label": "Preparation Study Days",
        "min": 5,
        "max": 90,
        "step": 1,
        "default": 35,
        "unit": "days",
        "description": "Total number of dedicated study days before exam"
    },
}

def generate_synthetic_data(n_samples: int = 1200, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic student performance data with realistic collinearity,
    diminishing returns, and realistic variance.
    """
    np.random.seed(random_seed)

    # Base academic aptitude latent factor
    aptitude = np.random.normal(loc=0.0, scale=1.0, size=n_samples)

    # Study hours (partly driven by aptitude, skewed positive)
    study_hours = np.clip(
        np.random.normal(loc=4.5 + 0.8 * aptitude, scale=2.0, size=n_samples),
        0.5, 14.0
    ).round(1)

    # Attendance percentage (correlated with dedication)
    attendance = np.clip(
        np.random.normal(loc=80.0 + 5.0 * aptitude, scale=10.0, size=n_samples),
        45.0, 100.0
    ).round(1)

    # Assignment score (out of 100)
    assignments = np.clip(
        np.random.normal(loc=74.0 + 8.0 * aptitude + 1.2 * study_hours, scale=9.0, size=n_samples),
        25.0, 100.0
    ).round(1)

    # Previous exam marks (out of 100)
    prev_exam = np.clip(
        np.random.normal(loc=68.0 + 11.0 * aptitude, scale=11.0, size=n_samples),
        25.0, 100.0
    ).round(1)

    # Internal marks (out of 50)
    internal = np.clip(
        np.random.normal(
            loc=35.0 + 4.5 * aptitude + 0.08 * attendance + 0.06 * assignments,
            scale=5.0,
            size=n_samples
        ),
        10.0, 50.0
    ).round(1)

    # Study days before exam (5 to 75 days)
    study_days = np.clip(
        np.random.normal(loc=35 + 5 * aptitude, scale=14, size=n_samples),
        5, 80
    ).astype(int)

    # Exam marks calculation formula (realistic education physics):
    # Normalized weights summing realistically
    # Internal scaled to 100 for balance
    internal_norm = (internal / 50.0) * 100.0

    # Diminishing returns on study hours beyond 9 hours
    study_hours_effective = np.where(study_hours > 8.0, 8.0 + (study_hours - 8.0) * 0.45, study_hours)

    raw_exam_marks = (
        0.28 * prev_exam +
        0.24 * internal_norm +
        0.18 * assignments +
        1.75 * study_hours_effective +
        0.12 * attendance +
        0.08 * study_days +
        np.random.normal(loc=0.0, scale=3.2, size=n_samples) # Natural exam day noise
    )

    # Final bounded exam marks (out of 100)
    exam_marks = np.clip(raw_exam_marks, 15.0, 100.0).round(1)

    # Create Student IDs
    student_ids = [f"STU{1000 + i}" for i in range(n_samples)]

    df = pd.DataFrame({
        "student_id": student_ids,
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "assignment_score": assignments,
        "previous_exam_marks": prev_exam,
        "internal_marks": internal,
        "study_days": study_days,
        "exam_marks": exam_marks
    })

    return df

def load_or_create_dataset(csv_path: str = DEFAULT_CSV_PATH, force_regenerate: bool = False) -> pd.DataFrame:
    """Loads existing dataset or generates and saves a default one."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path) or force_regenerate:
        df = generate_synthetic_data()
        df.to_csv(csv_path, index=False)
        return df
    try:
        df = pd.read_csv(csv_path)
        # Validate required columns
        missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
        if missing:
            # Recreate if schema is incorrect
            df = generate_synthetic_data()
            df.to_csv(csv_path, index=False)
        return df
    except Exception:
        df = generate_synthetic_data()
        df.to_csv(csv_path, index=False)
        return df

def clean_and_validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Cleans dataframe: handles missing values, types, and clips outliers."""
    issues = []
    df_clean = df.copy()

    # Drop non-feature non-target columns if any
    null_counts = df_clean[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum()
    if null_counts.sum() > 0:
        issues.append(f"Imputed {null_counts.sum()} missing values using column medians.")
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    # Clip values to plausible bounds
    for col, meta in FEATURE_METADATA.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(meta["min"], meta["max"])

    if TARGET_COLUMN in df_clean.columns:
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].clip(0.0, 100.0)

    return df_clean, issues
