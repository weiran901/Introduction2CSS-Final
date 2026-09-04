# Qilin Xiaohongshu Recommendation Project

This repository contains a final project exploring recommendation behavior in the Qilin/Xiaohongshu dataset. The project includes exploratory data analysis of note-level metadata and a baseline learning-to-rank style click prediction model for recommendation candidates.

## Project Structure

```text
.
|-- analysis.ipynb                  # Notebook entry point for exploratory analysis
|-- newmodel.ipynb                  # Notebook entry point for model training/evaluation
|-- data/
|   `-- qilin_note_metadata_final.parquet
|-- src/
|   `-- qilin_recommender/
|       |-- analysis.py             # EDA report generation
|       |-- data.py                 # Data loading and request flattening
|       |-- metrics.py              # Ranking metrics
|       `-- model.py                # Baseline click prediction model
|-- requirements.txt
|-- pyproject.toml
|-- setup.cfg
|-- setup.py
`-- README.md
```

## Dataset

The project uses the public Hugging Face dataset `THUIR/Qilin`. Qilin separates recommendation train/test data as dataset configs rather than Hugging Face split names:

- `recommendation_train`
- `recommendation_test`
- `search_train`
- `search_test`
- `notes`
- `user_feat`
- `dqa`

The local file `data/qilin_note_metadata_final.parquet` is a prepared note metadata table with fields such as `note_idx`, `commercial_flag`, `note_type`, `content_length`, `image_num`, `video_duration`, `taxonomy1_id`, `imp_rec_num`, and `click_rec_num`.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Analysis

```bash
PYTHONPATH=src python -m qilin_recommender.analysis
```

This creates summary tables and figures under `reports/`:

- `reports/metadata_summary.csv`
- `reports/taxonomy_ctr.csv`
- `reports/figures/note_type_distribution.png`
- `reports/figures/prior_ctr_distribution.png`
- `reports/figures/top_taxonomies_by_impressions.png`

## Train the New Model

```bash
PYTHONPATH=src python -m qilin_recommender.model
```

For a faster test run, limit the number of request-level examples:

```bash
PYTHONPATH=src python -m qilin_recommender.model --max-train-requests 5000 --max-test-requests 2000
```

The training script downloads the required Qilin recommendation configs from Hugging Face if they are not already cached. It saves:

- `models/new_model.joblib`
- `reports/new_model_metrics.json`

## Model Overview

The model is a baseline logistic regression classifier that predicts whether a displayed recommendation candidate receives a click. Request-level recommendation logs are flattened into candidate-level rows, then joined with note metadata.

Features include:

- request/context features: query length, recent click count, recommendation position
- note metadata: note type, content length, image count, video duration, commercial flag
- popularity signals: recommendation impressions, recommendation clicks, historical CTR
- taxonomy category: `taxonomy1_id`

Evaluation reports ROC-AUC, average precision, click rate, and mean NDCG@10 across recommendation requests.

## Notes for GitHub

The `.gitignore` excludes the local virtual environment, generated reports, trained models, caches, and temporary notebook files. The current repository is designed so another user can clone it, install dependencies, and rerun the analysis/model pipeline from the source code.

If you are creating a new GitHub repository from this folder, add the source files and the prepared metadata file, but do not commit `.venv/`, `hf_cache/`, `reports/`, or `models/`.
