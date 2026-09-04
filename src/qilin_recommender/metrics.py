from __future__ import annotations

import numpy as np
import pandas as pd


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@K for one request."""
    if len(labels) == 0:
        return 0.0

    order = np.argsort(scores)[::-1][:k]
    ranked_labels = labels[order]
    gains = (2**ranked_labels - 1) / np.log2(np.arange(2, len(ranked_labels) + 2))
    dcg = gains.sum()

    ideal_order = np.argsort(labels)[::-1][:k]
    ideal_labels = labels[ideal_order]
    ideal_gains = (2**ideal_labels - 1) / np.log2(np.arange(2, len(ideal_labels) + 2))
    ideal_dcg = ideal_gains.sum()
    return float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def mean_ndcg_at_k(
    frame: pd.DataFrame,
    label_col: str,
    score_col: str,
    group_col: str = "request_idx",
    k: int = 10,
) -> float:
    """Average NDCG@K across request groups."""
    values = []
    for _, group in frame.groupby(group_col):
        labels = group[label_col].to_numpy()
        scores = group[score_col].to_numpy()
        values.append(ndcg_at_k(labels, scores, k=k))
    return float(np.mean(values)) if values else 0.0

