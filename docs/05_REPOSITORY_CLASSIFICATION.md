# Repository Classification

This stage converts clustered functionality ids into repository-level vectors and trains or evaluates a binary classifier.

## CLI entrypoints

```bash
sastllm classify --mode search
sastllm classify --mode train
sastllm classify --mode test
sastllm-inspect-distribution --repository-id <id>
sastllm-inspect-timeseries --repository-id <id>
sastllm train
sastllm test
```

Implementation paths:

- Pipeline wrapper: `src/scripts/pipelines.py::classify_repositories`
- Classification service: `src/sastllm/classification/service.py::RepositoryClassificationService`
- Classification config: `src/sastllm/classification/config.py`
- Repository encoders: `src/sastllm/classification/encoders.py`
- Dataset assembly: `src/sastllm/classification/data.py`
- Metrics: `src/sastllm/classification/metrics.py`
- Inspection scripts: `src/scripts/inspect_repository_distribution_encoding.py`, `src/scripts/inspect_repository_timeseries_encoding.py`
- ML datasets/training/models: `src/sastllm/ml/datasets.py`, `src/sastllm/ml/training.py`, `src/sastllm/ml/models/`

## High-level wrappers

The high-level wrappers are compact aliases:

```text
sastllm train -> cluster --mode train -> classify --mode train
sastllm test  -> cluster --mode test  -> classify --mode test
```

They do not load data, split data, generate functionalities, or create embeddings.

## Configuration

Current `configs/classification.yaml`:

```yaml
classification:
  train:
    save_model_dir: "models/classification/trained_models"
    params:
      k: 10661
      lr: 0.0005
      weight_decay: 0.0005
      l1_param: 0.005
      epochs: 30
      batch_size: 64
      seed: 42

  test:
    load_model_dir: "models/classification/trained_models/model_20260426_204737"
    params:
      k: 10661
      lr: 0.0005
      weight_decay: 0.0005
      l1_param: 0.005
      batch_size: 64
      seed: 42
```

The YAML key `l1_param` is accepted as an alias for the runtime field `l1_lambda`.

## Data fetch

`RepositoryClassificationService` fetches repositories through:

```python
RepositoryManager.get_repositories_with_cluster_ids()
```

The query:

- starts from `repositories`
- left joins `files`, `snippets`, and `functionalities`
- filters `repositories.processed == true`
- optionally filters by `repositories.split`
- counts functionality cluster ids per repository

The resulting DTO shape is:

```text
repository_id
label
data = {cluster_id: count}
ordered_functionalities = [{functionality_id, cluster_id}, ...]
```

Before encoding, every label that is not exactly `benign` is rewritten to `malicious`.

## Repository vectorization

`ClusterDistributionEncoder` creates a fixed-width vector of length `k`.

For a repository with cluster counts:

```text
{7: 3, 19: 1}
```

and total count `4`, the raw vector has:

```text
x[7] = 0.75
x[19] = 0.25
```

`encode_repos()` then normalizes the entire feature matrix by its global L2 norm.

Cluster ids are zero-based. A database `cluster_id` of `0` maps directly to feature column `0`.

`OrderedFunctionalityTimeSeriesEncoder` creates a sequence-like feature matrix:

```text
shape = [num_functionalities, k]
```

The encoder:

1. sorts functionalities by `functionality_id` in ascending order
2. reads each functionality `cluster_id`
3. creates one one-hot row per functionality

For example, ordered cluster ids:

```text
5 -> 0 -> 5 -> 4
```

with `k = 6` become:

```text
[
  [0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 1, 0],
]
```

This representation preserves functionality order and is suitable for sequence or time-series model experiments.

Binary labels come from `configs/split.yaml`:

```yaml
binary_labels:
  - 0: "benign"
  - 1: "malicious"
```

## Encoder inspection scripts

Two scripts inspect a single repository from PostgreSQL, print its files/snippets/functionalities, and show exactly how the repository is encoded.

Distribution encoder:

```bash
sastllm-inspect-distribution --repository-id 123
sastllm-inspect-distribution --repository-name owner/repo
```

This prints:

- repository metadata
- files and snippets
- functionality ids, tags, descriptions, and cluster ids
- raw cluster counts
- non-zero `ClusterDistributionEncoder` feature values

Time-series encoder:

```bash
sastllm-inspect-timeseries --repository-id 123
sastllm-inspect-timeseries --repository-name owner/repo
```

This prints:

- the same repository/file/functionality tree
- the encoded feature shape
- the preserved timestep order
- each timestep's `functionality_id`, source lines, `cluster_id`, and active one-hot column

Useful flags:

```bash
--num-clusters 10661
--show-full-vector
--max-description-chars 240
--max-sequence-length 512
--truncation first
--truncation last
```

`--max-sequence-length` applies only to the time-series script. If a repository has more clustered functionalities than this limit, `--truncation first` keeps the earliest functionality ids and `--truncation last` keeps the latest functionality ids. If the sequence is shorter than the limit, the remaining rows are zero-padding.

## Training flow

## Search flow

`classify --mode search` reads `classification.search.grid_search` from `configs/classification.yaml`, builds every parameter combination, trains one model per combination, and writes:

- search model directories under `classification.search.save_model_dir`
- per-run accuracy plots under `classification.search.save_plots_dir`
- `search_summary.json` under `classification.search.save_plots_dir`
- `train_predictions.json`, `train_metrics.json`, `val_predictions.json`, and `val_metrics.json` inside each search model directory

Each plot contains training and validation accuracy over epochs.

`classify --mode train`:

1. Loads train params and `save_model_dir`.
2. Seeds Lightning and workers.
3. Fetches and encodes the full processed repository dataset.
4. Builds a train/validation/test data module.
5. Splits train repositories into train and validation with validation size `0.1`.
6. Uses stratified train/validation splitting.
7. Trains the configured Lightning model. The current default is `MLPRepositoryClassifier`.
8. Monitors `val_acc`.
9. Uses early stopping with patience `10`.
10. Saves the best checkpoint as `best.ckpt`.
11. Writes `config.json` and `meta.json`.
12. Evaluates the saved model on the train split.

Model output directory:

```text
models/classification/trained_models/model_<YYYYMMDD_HHMMSS>/
```

## Test flow

`classify --mode test`:

1. Loads `classification.test.load_model_dir`.
2. Loads `best.ckpt` from that directory.
3. Runs Lightning test on the data module.
4. Predicts repository classes for the test split.
5. Writes `test_predictions.json`.
6. Computes and writes `test_metrics.json`.

## Metrics

The evaluator computes:

- accuracy
- macro precision/recall/F1
- weighted precision/recall/F1
- per-class support, precision, recall, F1, TP, FP, FN
- confusion matrix
- macro and weighted AUC when possible
- per-class AUC and ROC curve data when possible

Predictions are stored as:

```json
{
  "12": {
    "label": "benign",
    "prediction": "malicious"
  }
}
```

## Output files

| File | Meaning |
| --- | --- |
| `best.ckpt` | saved Lightning checkpoint |
| `config.json` | serialized classifier config |
| `meta.json` | save timestamp, source checkpoint, monitored metric |
| `train_predictions.json` | optional persisted train predictions |
| `train_metrics.json` | train evaluation metrics |
| `test_predictions.json` | test predictions |
| `test_metrics.json` | test evaluation metrics |

## Common failure points

- No repositories are returned if `repositories.processed` is false.
- Train/validation split fails if there are too few examples per class.
- `best.ckpt` must exist for test mode.
- Classification `k` must match clustering `k`.
- Repository ids are converted to dataset indices with `repository_id - 1`; this assumes dense, 1-based repository ids.
