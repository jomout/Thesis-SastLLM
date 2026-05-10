# Repository Classification

This stage converts clustered functionality ids into repository-level vectors and trains or evaluates a binary classifier.

## CLI entrypoints

```bash
sastllm classify --mode train
sastllm classify --mode test
sastllm train
sastllm test
```

Implementation paths:

- Pipeline wrapper: `src/scripts/pipelines.py::classify_repositories`
- Classifier orchestrator: `src/sastllm/ml/repository_classifier.py::RepositoryClassifier`
- Repository encoder: `src/sastllm/utils/repository_encoder.py::RepositoryEncoder`
- Dataset/model classes: `src/sastllm/ml/dataset.py`, `src/sastllm/ml/model.py`

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

Important implementation detail: `ClassifierConfig` defines the L1 field as `l1_lambda`, while the YAML currently uses `l1_param`. With Pydantic's default behavior, `l1_param` is not the configured field name. Unless aliases or extra-field behavior are changed elsewhere, the classifier's default `l1_lambda=0.001` may be used instead of the YAML value.

## Data fetch

`RepositoryClassifier` fetches repositories through:

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
```

Before encoding, every label that is not exactly `benign` is rewritten to `malicious`.

## Repository vectorization

`RepositoryEncoder` creates a fixed-width vector of length `k`.

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

Binary labels come from `configs/split.yaml`:

```yaml
binary_labels:
  - 0: "benign"
  - 1: "malicious"
```

## Training flow

`classify --mode train`:

1. Loads train params and `save_model_dir`.
2. Seeds Lightning and workers.
3. Fetches and encodes the full processed repository dataset.
4. Builds a train/validation/test data module.
5. Splits train repositories into train and validation with validation size `0.1`.
6. Uses stratified train/validation splitting.
7. Trains a Lightning `CodeModel`.
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
