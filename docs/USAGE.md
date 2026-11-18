# Usage

This document lists the SAST-LLM CLI commands and common execution flows.

Full experiment runbook: [09_FULL_PIPELINE_RUNBOOK.md](./09_FULL_PIPELINE_RUNBOOK.md)

## CLI

```bash
argus --help
```

The CLI loads `.env` and configures logging before running a command.

## Commands

### Dataset ingestion

```bash
argus load
```

Loads the dataset from `configs/base.yaml -> paths.dataset`, discovers supported source files, chunks them, and inserts repository/file/snippet rows into PostgreSQL.

Reference: [01_DATA_INGESTION_AND_CHUNKING.md](./01_DATA_INGESTION_AND_CHUNKING.md)

### Dataset download

```bash
argus download_benign_dataset \
  --split train \
  --repositories-per-language 100
```

Downloads a repository-balanced benign subset from CodeSearchNet's Go, Java, JavaScript, PHP, Python, and Ruby configurations. The command selects the same number of unique repositories for every language, up to the requested maximum. If the least represented language has fewer eligible repositories, all languages are reduced to that count.

Selection is deterministic with the default seed of `42`. Pass `--seed` to choose another deterministic sample. Repositories present in more than one CodeSearchNet language configuration are excluded to prevent duplicate repository identities and later train/test leakage.

The downloader writes repositories beneath `configs/base.yaml -> paths.dataset/benign` using collision-safe `owner__repository` directory names. It refuses to append to an existing selected repository directory, because doing so would duplicate functions from an earlier export.

Supported options:

| Option                        | Default | Meaning                                           |
| ----------------------------- | ------- | ------------------------------------------------- |
| `--split train\|test`         | `train` | CodeSearchNet split                               |
| `--repositories-per-language` | `100`   | maximum unique repositories selected per language |
| `--seed`                      | `42`    | deterministic repository-selection seed           |

### Functionality generation

```bash
argus generate_functionalities
```

Uses the configured LLM to generate functionality descriptions and normalized tags for unprocessed snippets.

Reference: [02_FUNCTIONALITY_GENERATION.md](./02_FUNCTIONALITY_GENERATION.md)

### Batch API functionality generation

```bash
argus generate_functionalities_batch_api
```

Creates OpenAI Batch API JSONL request files, submits them, polls until completion, downloads output/error files, and imports successful functionality results.

Batch files are submitted and monitored concurrently. The concurrency limit is
configured with `batch_functionality.max_concurrent_batches` in
`configs/llms.yaml` and defaults to `3`. Any failed file prevents result import
from being reported as a successful complete wave.

The command uses the validated `batch_functionality` section in `configs/llms.yaml`. It currently targets `gpt-5.6-luna` through `/v1/responses`, imports successful outputs into PostgreSQL, and writes its artifacts to:

```text
api_batches/
batch_results/
```

Move completed artifacts before starting another run; the command refuses to mix or overwrite existing JSONL request and result files.

### Cached functionality import

```bash
argus load_cache_functionalities /path/to/cached_functionalities
```

Loads JSON files named like `functionalities_<snippet_id>.json`, inserts functionality rows, and marks snippets processed.

### Functionality embeddings

```bash
argus embed
```

Creates missing embeddings for `functionalities.tag` in the Qdrant collection derived from `split.model_name`. Existing functionality point ids are skipped.

### Split

```bash
argus split
```

Assigns train/test split values to repositories and updates Qdrant point payloads.

Reference: [03_EMBEDDING_AND_SPLITTING.md](./03_EMBEDDING_AND_SPLITTING.md)

### Clustering

```bash
argus cluster --mode search --config-path configs/clustering.yaml
argus cluster --mode train --config-path configs/clustering.yaml
argus cluster --mode test --config-path configs/clustering.yaml
```

Modes:

| Mode     | Behavior                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------ |
| `search` | compare candidate K values using cohesion, separation, silhouette, CH, and inertia; save reports, model, and plots |
| `train`  | fit MiniBatchKMeans, write a full quality report, and assign train cluster ids                                     |
| `test`   | load trained clusterer and assign test cluster ids                                                                 |

Reference: [04_FUNCTIONALITY_CLUSTERING.md](./04_FUNCTIONALITY_CLUSTERING.md)

Research workflow: [08_CLUSTERING_RESEARCH_GUIDE.md](./08_CLUSTERING_RESEARCH_GUIDE.md)

### Classification

```bash
argus classify --mode search --config-path configs/classification.mlp.yaml
argus classify --mode train --config-path configs/classification.mlp.yaml
argus classify --mode test --config-path configs/classification.mlp.yaml
```

All three modes require `--config-path`. Pass `configs/classification.transformer.yaml` to run the Transformer instead. Each file selects its matching checkpoint under `selected_models/` for test mode and has its own search and training output directories. The `argus train` and `argus test` wrappers require `--config-path` for classification and `--clustering-config-path` for clustering.

Modes:

| Mode     | Behavior                                                                   |
| -------- | -------------------------------------------------------------------------- |
| `search` | grid-search classifier params, save models, plot train/validation accuracy |
| `train`  | train classifier, save checkpoint, write train metrics                     |
| `test`   | load checkpoint, run test, write predictions and metrics                   |

Reference: [05_REPOSITORY_CLASSIFICATION.md](./05_REPOSITORY_CLASSIFICATION.md)

Classifier architectures are defined under the top-level `models` registry and selected through `classification.<mode>.model`: `mlp`, `lstm`, or `transformer`. Set `models.transformer.input_encoding` to `"ordered_tokens"` for order-aware functionality sequences or `"cluster_distribution"` for aggregate cluster-frequency attention. A practical starting profile is `embedding_dim: 128`, `num_heads: 4`, `num_layers: 2`, `feedforward_dim: 256`, `pooling: "mean"`, and `max_sequence_length: 256`.

### SHAP explanations

Explain the classifier selected by `classification.test`:

```bash
argus explain-shap --config-path configs/classification.mlp.yaml \
  --class malicious \
  --split test \
  --background-split train \
  --background-samples 200 \
  --explanation-samples 16 \
  --top-features 30 \
  --output-dir plots/shap
```

The command reuses the test-mode model profile, cluster count, label mapping, encoder, dataset splits, and `load_model_dir`. It verifies the classifier manifest and checksum contract before computing explanations. `GradientExplainer` currently supports the MLP and the Transformer with `input_encoding: "cluster_distribution"`; ordered-token LSTM and Transformer inputs are discrete and are rejected with an actionable error.

The output directory contains `shap_values.npy`, the explained input rows, repository ids and labels, `meta.json`, a global summary plot, a signed-mean plot, and a waterfall plot for the first explained repository. Select the classifier with the required `--config-path` and control execution with `--device cpu|cuda|auto`.

Use `--cluster-descriptions PATH` to add titles from a [JSON mapping for the fitted clusterer](./05_REPOSITORY_CLASSIFICATION.md#shap-explanations).

### Confusion-matrix plots

Evaluate the classifier selected by `classification.test` and plot its confusion matrix:

```bash
argus confusion-matrix --config-path configs/classification.mlp.yaml \
  --split test \
  --log-scale \
  --format png \
  --output-dir plots/confusion_matrix
```

The command uses the configured model profile, encoder, labels, split, and `load_model_dir`, and verifies the classifier bundle before inference. It writes count and row-normalized plots plus a JSON file containing the underlying matrices and headline metrics. Use a distinct output directory for each model comparison, such as `plots/confusion_matrix/mlp` and `plots/confusion_matrix/transformer`.

### Repository encoding inspection

```bash
argus-inspect-distribution --config configs/classification.mlp.yaml --repository-id 123
argus-inspect-timeseries --config configs/classification.mlp.yaml --repository-id 123
```

These scripts load one repository from PostgreSQL, print its files/snippets/functionalities, and show how the repository is encoded.

Pass `--config` to read the cluster count from a classification YAML, or supply the count directly with `--num-clusters`.

Use `--repository-name <name>` instead of `--repository-id` when the repository name is easier to target.

Common flags:

| Flag                          | Meaning                                                         |
| ----------------------------- | --------------------------------------------------------------- |
| `--num-clusters 10661`        | override `classification.<mode>.params.k`                       |
| `--show-full-vector`          | print complete feature vectors instead of only non-zero entries |
| `--max-description-chars 240` | control functionality description truncation in printed output  |
| `--max-sequence-length 512`   | pad/truncate the time-series encoder output to a fixed length   |
| `--truncation first\|last`    | for long sequences, keep earliest or latest functionality ids   |

Reference: [05_REPOSITORY_CLASSIFICATION.md](./05_REPOSITORY_CLASSIFICATION.md#encoder-inspection-scripts)

### Pipeline wrappers

```bash
argus train --config-path configs/classification.mlp.yaml --clustering-config-path configs/clustering.yaml
argus test --config-path configs/classification.mlp.yaml --clustering-config-path configs/clustering.yaml
```

Current wrapper behavior:

```text
train -> cluster --mode train -> classify --mode train
test  -> cluster --mode test  -> classify --mode test
```

These wrappers do not execute loading, functionality generation, embedding creation, or splitting.

## Typical full workflow

```bash
argus load
argus generate_functionalities
argus embed
argus split
argus cluster --mode train --config-path configs/clustering.yaml
argus classify --mode train --config-path configs/classification.mlp.yaml
argus cluster --mode test --config-path configs/clustering.yaml
argus classify --mode test --config-path configs/classification.mlp.yaml
```

## Cached functionality workflow

```bash
argus load
argus load_cache_functionalities /path/to/cached_functionalities
argus embed
argus split
argus cluster --mode train --config-path configs/clustering.yaml
argus classify --mode train --config-path configs/classification.mlp.yaml
```

## Model reuse workflow

```bash
argus cluster --mode test --config-path configs/clustering.yaml
argus classify --mode test --config-path configs/classification.mlp.yaml
```

Use this when:

- `configs/clustering.yaml` points to the trained clusterer
- the classification YAML passed with `--config-path` points to the trained classifier directory
- test embeddings exist and have Qdrant payload `split=test`

## Output locations

| Location                                                                                       | Content                                                                |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `cache/functionalities-<llm_type>/`                                                            | parsed functionality caches                                            |
| `api_batches/`                                                                                 | Batch API request JSONL files                                          |
| `batch_results/`                                                                               | downloaded Batch API outputs/errors                                    |
| `models/clustering/trained_models/`                                                            | clustering joblib/ONNX bundles, manifests, and quality reports         |
| `models/classification/trained_models/`                                                        | classifier checkpoints, weights, ONNX, manifests, configs, and metrics |
| `models/clustering/searching_models/clusterers_<n>_<timestamp>/clusterer_<n>_<k>_<timestamp>/` | clustering search model, reports, CSV, and plot                        |
