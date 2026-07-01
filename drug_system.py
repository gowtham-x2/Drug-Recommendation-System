from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
DATASET_PATH = DATA_DIR / "drug_recommendation_sample.csv"
MODEL_PATH = MODEL_DIR / "drug_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.png"
FEATURE_NAMES = ["age", "sex", "blood_pressure", "cholesterol", "na_to_k"]
TARGET_NAME = "drug"
DRUG_LABELS = ["DrugA", "DrugB", "DrugC", "DrugX", "DrugY"]
RANDOM_STATE = 42


def ensure_directories() -> None:
    for directory in (DATA_DIR, MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _assign_drug(age: int, sex: str, blood_pressure: str, cholesterol: str, na_to_k: float) -> str:
    if na_to_k >= 22:
        return "DrugY"
    if blood_pressure == "HIGH" and cholesterol == "HIGH":
        return "DrugA" if age < 50 else "DrugB"
    if blood_pressure == "LOW":
        return "DrugC"
    if sex == "F" and cholesterol == "HIGH":
        return "DrugX"
    return "DrugB" if age >= 55 else "DrugX"


def generate_dataset(
    csv_path: Path = DATASET_PATH,
    n_samples: int = 3000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    ensure_directories()
    rng = np.random.default_rng(random_state)

    ages = rng.integers(18, 80, size=n_samples)
    sexes = rng.choice(["F", "M"], size=n_samples, p=[0.5, 0.5])
    blood_pressures = rng.choice(["LOW", "NORMAL", "HIGH"], size=n_samples, p=[0.2, 0.45, 0.35])
    cholesterol_levels = rng.choice(["NORMAL", "HIGH"], size=n_samples, p=[0.55, 0.45])
    na_to_k_values = np.round(rng.normal(loc=17, scale=5.5, size=n_samples).clip(6, 38), 2)

    drugs: list[str] = []
    for age, sex, blood_pressure, cholesterol, na_to_k in zip(
        ages, sexes, blood_pressures, cholesterol_levels, na_to_k_values
    ):
        drug = _assign_drug(int(age), str(sex), str(blood_pressure), str(cholesterol), float(na_to_k))
        if rng.random() < 0.06:
            alternatives = [label for label in DRUG_LABELS if label != drug]
            drug = str(rng.choice(alternatives))
        drugs.append(drug)

    frame = pd.DataFrame(
        {
            "age": ages,
            "sex": sexes,
            "blood_pressure": blood_pressures,
            "cholesterol": cholesterol_levels,
            "na_to_k": na_to_k_values,
            TARGET_NAME: drugs,
        }
    )
    frame.to_csv(csv_path, index=False)
    return frame


def load_or_create_dataset(csv_path: Path = DATASET_PATH) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return generate_dataset(csv_path=csv_path)


def _plot_confusion_matrix(matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Drug Recommendation Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_and_save(
    csv_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    confusion_matrix_path: Path = CONFUSION_MATRIX_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_directories()
    frame = load_or_create_dataset(csv_path)
    x = frame[FEATURE_NAMES]
    y = frame[TARGET_NAME]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["sex", "blood_pressure", "cholesterol"],
            ),
            ("numeric", StandardScaler(), ["age", "na_to_k"]),
        ],
        sparse_threshold=0,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    class_labels = list(pipeline.named_steps["model"].classes_)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(frame)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "class_labels": class_labels,
    }

    matrix = confusion_matrix(y_test, predictions, labels=class_labels)
    artifact = {
        "pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "class_labels": class_labels,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot_confusion_matrix(matrix, class_labels, confusion_matrix_path)
    return artifact, metrics


def load_artifact(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        train_and_save(model_path=model_path)
    return joblib.load(model_path)


def load_metrics(metrics_path: Path = METRICS_PATH) -> dict[str, Any]:
    if not metrics_path.exists():
        train_and_save(metrics_path=metrics_path)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def predict_single(patient: dict[str, Any], artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    artifact = artifact or load_artifact()
    ordered_values = {
        "age": int(patient["age"]),
        "sex": str(patient["sex"]),
        "blood_pressure": str(patient["blood_pressure"]),
        "cholesterol": str(patient["cholesterol"]),
        "na_to_k": float(patient["na_to_k"]),
    }
    frame = pd.DataFrame([ordered_values])
    predicted_drug = str(artifact["pipeline"].predict(frame)[0])
    probabilities = artifact["pipeline"].predict_proba(frame)[0]
    probability_map = {
        label: float(probability)
        for label, probability in zip(artifact["class_labels"], probabilities)
    }
    return {
        "recommended_drug": predicted_drug,
        "confidence": float(max(probability_map.values())),
        "probabilities": probability_map,
    }


def predict_batch(frame: pd.DataFrame, artifact: dict[str, Any] | None = None) -> pd.DataFrame:
    artifact = artifact or load_artifact()
    batch = frame.copy()
    predictions = artifact["pipeline"].predict(batch[artifact["feature_names"]])
    probabilities = artifact["pipeline"].predict_proba(batch[artifact["feature_names"]])
    batch["recommended_drug"] = predictions
    batch["confidence"] = probabilities.max(axis=1)
    return batch
