# Configuration

This document is the short configuration overview. For the full field-by-field reference, see [07_CONFIGURATION_REFERENCE.md](./07_CONFIGURATION_REFERENCE.md).

## Files

Configuration lives under `configs/`:

| File | Role |
| --- | --- |
| `base.yaml` | app name, logging, dataset path |
| `llms.yaml` | LLM provider and model for snippet functionality generation |
| `split.yaml` | embedding model, train/test ratios, binary labels |
| `clustering.yaml` | clustering search/train/test parameters |
| `classification.yaml` | classifier train/test directories and hyperparameters |
| `languages.yaml` | Tree-sitter grammar sources and supported suffixes |
| `important_nodes.yaml` | AST node types used as chunking breakpoints |
| `comment_nodes.yaml` | comment/decorator node types |

## Current core values

Dataset path:

```yaml
paths:
  dataset: ".dataset/thesis_dataset"
```

Logging:

```yaml
log:
  level: INFO
  file: logs/sastllm.log
  file_level: DEBUG
  max_bytes: 10485760
  backup_count: 5
```

Console logs use `log.level`. The rotating JSON-lines file uses `log.file_level`, so the default setup keeps normal console output readable while retaining detailed debugging context in `logs/sastllm.log`.

Log-level intent:

| Level | Content |
| --- | --- |
| `DEBUG` | batch iteration, paths, shapes, parser details, and configuration loading |
| `INFO` | pipeline lifecycle, model construction, data counts, metrics, and persisted artifacts |
| `WARNING` | recoverable anomalies such as truncation, empty sequences, unreliable silhouette, or class-imbalance overcorrection |
| `ERROR` | failed I/O, invalid operational state, unsupported configuration, and exceptions at pipeline boundaries |

LLM:

```yaml
models:
  snippet_processor:
    host: "google"
    name: "gemini-2.5-flash"
```

Embedding model and split:

```yaml
split:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  training:
    ratio: 0.7
  testing:
    ratio: 0.3
```

Clustering:

```yaml
clustering:
  evaluation:
    sample_size: 100000
    silhouette_sample_size: 5000
    silhouette_samples_per_cluster: 5
    silhouette_metric: "euclidean"
    random_state: 42
    elbow_window_factor: 2.0
    max_silhouette_singleton_fraction: 0.5
  train:
    k: 10661
  test:
    load_model_file: "models/clustering/trained_models/clusterer_k_10661.joblib"
```

Classification:

```yaml
models:
  mlp:
    hidden_dims: [512, 256]
    dropout: 0.2
  lstm:
    embedding_dim: 128
    hidden_dim: 128
    num_layers: 3
    dropout: 0.2
    bidirectional: false
    pooling: "last"
    max_sequence_length: 512
    truncation: "first"
  transformer:
    embedding_dim: 128
    num_layers: 2
    num_heads: 4
    feedforward_dim: 256
    dropout: 0.2
    pooling: "mean"
    max_sequence_length: 256
    truncation: "first"

classification:
  search:
    model: mlp
    save_model_dir: "models/classification/searching_models"
    save_plots_dir: "plots/classification/searching"
  train:
    model: lstm
    save_model_dir: "models/classification/trained_models"
    params:
      use_weighted_sampler: true
      use_class_weights: false
  test:
    model: lstm
    load_model_dir: "models/classification/trained_models/model_20260426_204737"
```

## Environment variables

The CLI loads `.env` at startup. Common variables are:

```text
GOOGLE_API_KEY
OPENAI_API_KEY
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_DB
POSTGRES_HOST
POSTGRES_PORT
ENDPOINT_URL
ACCESS_TOKEN
```

Use the API key for the provider selected in `configs/llms.yaml`.

## Important notes

- Only `snippet_processor` is currently supported by the LLM factory.
- Exceptions logged at pipeline and external-service boundaries include traceback context in the rotating JSON log.
- The Qdrant collection name is derived from `split.model_name` by replacing `/` with `_`.
- Clustering search validates candidate K values with normalized inertia, cohesion, separation, Calinski-Harabasz, sampled silhouette, and cluster-size health.
- Full silhouette is not computed for the complete embedding population; `clustering.evaluation` controls a seeded reservoir and cluster-stratified silhouette sample, and records reliability metadata.
- The classifier `k` should match the clustering `k`.
- `classification.yaml` may use `l1_param`; the classification config maps it to `l1_lambda`.
- `use_weighted_sampler` controls training-batch rebalancing.
- `use_class_weights` controls weighted `CrossEntropyLoss`.
- `classification.<mode>.model` selects a strict profile from the top-level `models` registry.
- `model: mlp` uses aggregate cluster-distribution vectors.
- `model: lstm` uses ordered functionality-cluster token sequences plus a learned embedding layer.
- `model: transformer` uses ordered functionality-cluster tokens, learned positional embeddings, and padding-masked self-attention.
- For LSTM models, `embedding_dim` controls the learned cluster-token embedding width, `max_sequence_length` controls sequence padding/truncation, and `pooling` may be `last`, `mean`, or `max`.
- For Transformer models, `embedding_dim` must be divisible by `num_heads`; `feedforward_dim` sets the encoder feed-forward width. Start with `max_sequence_length: 256` because attention cost grows quadratically with sequence length.
