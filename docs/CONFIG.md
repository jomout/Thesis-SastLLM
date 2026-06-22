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
- The Qdrant collection name is derived from `split.model_name` by replacing `/` with `_`.
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
