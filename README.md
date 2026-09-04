# Investigating a Recurring Click-Through Rate Anomaly in Xiaohongshu Recommendations

This repository contains the code and materials for a final project on recommendation behavior in Xiaohongshu, using the public Qilin dataset. The project investigates a recurring click-through rate (CTR) dip at the sixth recommendation position within repeated seven-item blocks.

The project moves from an initial commercial-exposure explanation toward a candidate-selection explanation: commercial content helps explain the pattern, but it does not fully account for the recurring engagement penalty.

## Research Question

CTR usually declines as recommendation rank gets lower. In the Qilin recommendation logs, however, the sixth position shows an unusually sharp CTR drop. The project asks:

> Why does the sixth slot within each seven-position recommendation block receive lower engagement?

The Qilin paper suggests that some positions may have a higher probability of exposing commercial notes. This project tests that idea and then examines whether the anomaly remains after controlling for observable note, topic, and user-context factors.

## Main Hypotheses

1. Commercial exposure hypothesis: the sixth slot has lower CTR because it contains more commercial content.
2. Position-structure hypothesis: the penalty is not limited to absolute rank 6, but recurs at positions 6, 13, 20, 27, and so on.
3. Candidate-selection hypothesis: even after adjusting for position and observable context, the sixth slot receives candidates with weaker out-of-sample click performance.

## Data

The project uses the Hugging Face dataset `THUIR/Qilin`, especially the recommendation logs and note metadata. Qilin stores train/test recommendation data as separate dataset configs:

- `recommendation_train`
- `recommendation_test`
- `notes`
- `user_feat`

The local prepared metadata file is:

```text
data/qilin_note_metadata_final.parquet
```

It includes fields such as `note_idx`, `commercial_flag`, `note_type`, `content_length`, `image_num`, `video_duration`, `taxonomy1_id`, `imp_rec_num`, and `click_rec_num`.

## Methodology

### 1. Commercial Flag Analysis

Notes are classified using `commercial_flag`:

- commercial content: `commercial_flag != 0`
- non-commercial content: `commercial_flag == 0`

Initial evidence shows that commercial content is strongly concentrated at the sixth position. The same pattern appears repeatedly at positions 6, 13, 20, 27, and later slots, suggesting a recurring seven-position serving structure.

### 2. Seven-Position Structure

Each recommendation position is converted into:

- block: the group of seven positions the item belongs to
- within-block position: the item's location inside that seven-position block

For example:

```text
Block 1: 1, 2, 3, 4, 5, 6, 7
Block 2: 8, 9, 10, 11, 12, 13, 14
Block 3: 15, 16, 17, 18, 19, 20, 21
```

Positions 6, 13, and 20 all correspond to within-block position 6.

### 3. Logistic Regression Models

The main regression tests whether the sixth within-block position has lower click probability:

```text
Click ~ C(Block) + C(WithinBlockPosition)
```

Additional models add controls for commercial exposure, content type/format, topic, and user context. If these controls fully explained the anomaly, the sixth-slot coefficient would move toward zero and the odds ratio would move toward one.

In the presentation results, the sixth-slot odds ratio remains below one after controls, which means observable controls do not fully eliminate the penalty.

### 4. Cross-Fitted Candidate Quality

The revised approach estimates expected click probability from position and context, then measures each candidate's performance relative to that expectation:

```text
adjusted_performance = click - expected_click_probability
```

Cross-fitting is used so that each observation is evaluated by a model that was not trained on that same observation. This helps separate genuine candidate quality from in-sample overfitting.

The sixth slot shows weaker position-adjusted, out-of-sample performance than neighboring positions, supporting the candidate-selection explanation.

## Key Findings

- CTR declines with rank overall, but the sixth slot shows an unusually sharp dip.
- Commercial content is concentrated at the sixth slot and recurs every seven positions.
- Removing commercial content does not eliminate the CTR dip.
- Logistic regression controls reduce but do not fully explain the sixth-slot penalty.
- Cross-fitted adjusted performance suggests that sixth-slot candidates are weaker even after accounting for position and context.

## Conclusion

Commercial exposure partially explains the recurring sixth-slot CTR penalty, but it is not the whole story. The anomaly may involve the candidate-selection or serving mechanism itself, not only ad exposure. A practical next step would be to audit these recurring slots and run A/B tests with alternative candidate-selection strategies.

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

## Train the Baseline Model

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

## Reference

Chen, J., Dong, Q., Li, H., He, X., Gao, Y., Cao, S., Wu, Y., Yang, P., Xu, C., Hu, Y., Ai, Q., & Liu, Y. (2025). Qilin: A multimodal information retrieval dataset with APP-level user sessions. In *Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 3670-3680). Association for Computing Machinery. https://doi.org/10.1145/3726302.3730279
