from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data import DEFAULT_METADATA_PATH, load_note_metadata


def run_analysis(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    output_dir: str | Path = "reports",
) -> dict:
    """Run note-metadata EDA and save summary tables/figures."""
    output_path = Path(output_dir)
    figure_path = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    notes = load_note_metadata(metadata_path)

    summary = {
        "num_notes": int(len(notes)),
        "num_columns": int(notes.shape[1]),
        "num_taxonomy1": int(notes["taxonomy1_id"].nunique(dropna=True)),
        "median_impressions": float(notes["imp_rec_num"].median()),
        "median_clicks": float(notes["click_rec_num"].median()),
        "mean_prior_ctr": float(notes["prior_rec_ctr"].mean(skipna=True)),
    }

    pd.Series(summary, name="value").to_csv(output_path / "metadata_summary.csv")

    taxonomy_summary = (
        notes.groupby("taxonomy1_id", dropna=False)
        .agg(
            notes=("note_idx", "count"),
            impressions=("imp_rec_num", "sum"),
            clicks=("click_rec_num", "sum"),
        )
        .reset_index()
    )
    taxonomy_summary["ctr"] = taxonomy_summary["clicks"] / taxonomy_summary["impressions"]
    taxonomy_summary.sort_values("impressions", ascending=False).to_csv(
        output_path / "taxonomy_ctr.csv",
        index=False,
    )

    _plot_note_type_distribution(notes, figure_path / "note_type_distribution.png")
    _plot_ctr_distribution(notes, figure_path / "prior_ctr_distribution.png")
    _plot_top_taxonomies(taxonomy_summary, figure_path / "top_taxonomies_by_impressions.png")

    print("Analysis complete.")
    print(f"Summary: {summary}")
    print(f"Reports saved to: {output_path.resolve()}")
    return summary


def _plot_note_type_distribution(notes: pd.DataFrame, output_file: Path) -> None:
    counts = notes["note_type"].fillna("missing").value_counts().sort_index()
    ax = counts.plot(kind="bar", color="#4c78a8", figsize=(8, 4))
    ax.set_title("Note Type Distribution")
    ax.set_xlabel("note_type")
    ax.set_ylabel("Number of notes")
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def _plot_ctr_distribution(notes: pd.DataFrame, output_file: Path) -> None:
    ctr = notes.loc[notes["imp_rec_num"] > 0, "prior_rec_ctr"].clip(0, 1)
    ax = ctr.plot(kind="hist", bins=60, color="#59a14f", figsize=(8, 4))
    ax.set_title("Historical Recommendation CTR Distribution")
    ax.set_xlabel("click_rec_num / imp_rec_num")
    ax.set_ylabel("Number of notes")
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def _plot_top_taxonomies(taxonomy_summary: pd.DataFrame, output_file: Path) -> None:
    top = taxonomy_summary.nlargest(15, "impressions").sort_values("impressions")
    ax = top.plot(
        kind="barh",
        x="taxonomy1_id",
        y="impressions",
        legend=False,
        color="#f28e2b",
        figsize=(9, 6),
    )
    ax.set_title("Top Taxonomies by Recommendation Impressions")
    ax.set_xlabel("Recommendation impressions")
    ax.set_ylabel("taxonomy1_id")
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for Qilin note metadata.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH), help="Path to note metadata parquet.")
    parser.add_argument("--output-dir", default="reports", help="Directory for generated reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(metadata_path=args.metadata, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

