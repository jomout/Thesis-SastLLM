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
classification:
  search:
    save_model_dir: "models/classification/searching_models"
    save_plots_dir: "plots/classification/searching"
  train:
    save_model_dir: "models/classification/trained_models"
  test:
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
