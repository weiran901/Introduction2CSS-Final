from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "qilin_note_metadata_final.parquet"

RECOMMENDATION_TRAIN_CONFIG = "recommendation_train"
RECOMMENDATION_TEST_CONFIG = "recommendation_test"


def load_note_metadata(path: str | Path = DEFAULT_METADATA_PATH) -> pd.DataFrame:
    """Load prepared note-level metadata and add popularity features."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Note metadata file not found: {metadata_path}. "
            "Expected data/qilin_note_metadata_final.parquet."
        )

    notes = pd.read_parquet(metadata_path)
    notes = notes.copy()

    for column in ["imp_rec_num", "click_rec_num"]:
        if column not in notes.columns:
            notes[column] = np.nan

    impressions = pd.to_numeric(notes["imp_rec_num"], errors="coerce")
    clicks = pd.to_numeric(notes["click_rec_num"], errors="coerce")
    notes["prior_rec_ctr"] = clicks.div(impressions.where(impressions > 0)).clip(0, 1)

    video_duration = pd.to_numeric(notes.get("video_duration", 0), errors="coerce")
    image_num = pd.to_numeric(notes.get("image_num", 0), errors="coerce")
    notes["has_video"] = (video_duration.fillna(0) > 0).astype(int)
    notes["has_images"] = (image_num.fillna(0) > 0).astype(int)
    return notes


def load_recommendation_split(
    config_name: str,
    split: str = "train",
    cache_dir: str | Path | None = None,
):
    """Load a Qilin recommendation config from Hugging Face.

    Qilin stores train/test recommendation data as separate configs, so the
    Hugging Face split is still named "train" for both configs.
    """
    return load_dataset(
        "THUIR/Qilin",
        config_name,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def flatten_recommendation_requests(
    requests: Iterable[dict],
    max_requests: int | None = None,
) -> pd.DataFrame:
    """Convert request-level recommendation logs into candidate-level rows."""
    rows: list[dict] = []

    for request_number, request in enumerate(requests):
        if max_requests is not None and request_number >= max_requests:
            break

        details = request.get("rec_result_details_with_idx") or []
        recent_clicked = request.get("recent_clicked_note_idxs") or []
        query = request.get("query") or ""

        for item in details:
            rows.append(
                {
                    "request_idx": request.get("request_idx"),
                    "session_idx": request.get("session_idx"),
                    "user_idx": request.get("user_idx"),
                    "query": query,
                    "query_length": len(query),
                    "recent_click_count": len(recent_clicked),
                    "note_idx": item.get("note_idx"),
                    "position": item.get("position"),
                    "clicked": int((item.get("click") or 0) > 0),
                    "liked": int((item.get("like") or 0) > 0),
                    "collected": int((item.get("collect") or 0) > 0),
                    "commented": int((item.get("comment") or 0) > 0),
                    "shared": int((item.get("share") or 0) > 0),
                    "page_time": item.get("page_time"),
                }
            )

    return pd.DataFrame(rows)


def build_modeling_table(
    recommendation_df: pd.DataFrame,
    note_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Join candidate rows to note metadata and keep the modeling columns."""
    if recommendation_df.empty:
        raise ValueError("No recommendation rows were available after flattening.")

    model_df = recommendation_df.merge(note_metadata, on="note_idx", how="left")
    model_df["taxonomy1_id"] = model_df["taxonomy1_id"].fillna("unknown")

    return model_df
