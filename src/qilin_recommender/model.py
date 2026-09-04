from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (
    DEFAULT_METADATA_PATH,
    RECOMMENDATION_TEST_CONFIG,
    RECOMMENDATION_TRAIN_CONFIG,
    build_modeling_table,
    flatten_recommendation_requests,
    load_note_metadata,
    load_recommendation_split,
)
from .metrics import mean_ndcg_at_k


NUMERIC_FEATURES = [
    "query_length",
    "recent_click_count",
    "position",
    "commercial_flag",
    "note_type",
    "content_length",
    "image_num",
    "video_duration",
    "imp_rec_num",
    "click_rec_num",
    "prior_rec_ctr",
    "has_video",
    "has_images",
]

CATEGORICAL_FEATURES = ["taxonomy1_id"]
TARGET = "clicked"


def make_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def load_training_frame(
    config_name: str,
    metadata_path: str | Path,
    max_requests: int | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    dataset = load_recommendation_split(config_name, cache_dir=cache_dir)
    candidates = flatten_recommendation_requests(dataset, max_requests=max_requests)
    notes = load_note_metadata(metadata_path)
    return build_modeling_table(candidates, notes)


def train_and_evaluate(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    model_output: str | Path = "models/new_model.joblib",
    metrics_output: str | Path = "reports/new_model_metrics.json",
    max_train_requests: int | None = None,
    max_test_requests: int | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    train_df = load_training_frame(
        RECOMMENDATION_TRAIN_CONFIG,
        metadata_path=metadata_path,
        max_requests=max_train_requests,
        cache_dir=cache_dir,
    )
    test_df = load_training_frame(
        RECOMMENDATION_TEST_CONFIG,
        metadata_path=metadata_path,
        max_requests=max_test_requests,
        cache_dir=cache_dir,
    )

    model = make_pipeline()
    model.fit(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df[TARGET])

    test_scores = predict_click_probabilities(model, test_df)
    test_df = test_df.copy()
    test_df["predicted_click_probability"] = test_scores

    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_click_rate": float(train_df[TARGET].mean()),
        "test_click_rate": float(test_df[TARGET].mean()),
        "roc_auc": _safe_roc_auc(test_df[TARGET], test_scores),
        "average_precision": float(average_precision_score(test_df[TARGET], test_scores)),
        "mean_ndcg_at_10": mean_ndcg_at_k(
            test_df,
            label_col=TARGET,
            score_col="predicted_click_probability",
            k=10,
        ),
    }

    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metrics_path = Path(metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Model training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {model_path.resolve()}")
    print(f"Metrics saved to: {metrics_path.resolve()}")
    return metrics


def predict_click_probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Return click probabilities with a stable explicit sigmoid calculation."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    logits = (
        np.einsum("ij,j->i", transformed, classifier.coef_.ravel())
        + classifier.intercept_[0]
    )
    logits = np.clip(logits, -500, 500)
    return 1 / (1 + np.exp(-logits))


def _safe_roc_auc(labels: pd.Series, scores) -> float | None:
    if labels.nunique() < 2:
        return None
    return float(roc_auc_score(labels, scores))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Qilin recommendation click model.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH), help="Path to note metadata parquet.")
    parser.add_argument("--model-output", default="models/new_model.joblib", help="Path for the trained model.")
    parser.add_argument("--metrics-output", default="reports/new_model_metrics.json", help="Path for metrics JSON.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--max-train-requests", type=int, default=None, help="Optional training request limit.")
    parser.add_argument("--max-test-requests", type=int, default=None, help="Optional test request limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(
        metadata_path=args.metadata,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
        max_train_requests=args.max_train_requests,
        max_test_requests=args.max_test_requests,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
