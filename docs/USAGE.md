# Usage

This document lists the SAST-LLM CLI commands and common execution flows.

## CLI

```bash
sastllm --help
```

The CLI loads `.env` and configures logging before running a command.

## Commands

### Dataset ingestion

```bash
sastllm load
```

Loads the dataset from `configs/base.yaml -> paths.dataset`, discovers supported source files, chunks them, and inserts repository/file/snippet rows into PostgreSQL.

Reference: [01_DATA_INGESTION_AND_CHUNKING.md](./01_DATA_INGESTION_AND_CHUNKING.md)

### Dataset download

```bash
sastllm download_benign_dataset
```

Downloads the benign CodeSearchNet-derived dataset using `src/scripts/download_dataset.py`.

### Functionality generation

```bash
sastllm generate_functionalities
```

Uses the configured LLM to generate functionality descriptions and normalized tags for unprocessed snippets.

Reference: [02_FUNCTIONALITY_GENERATION.md](./02_FUNCTIONALITY_GENERATION.md)

### Batch API functionality generation

```bash
sastllm generate_functionalities_batch_api
```

Creates OpenAI Batch API JSONL request files, submits them, polls until completion, and downloads raw output/error files.

Current output directories:

```text
api_batches_extra/
batch_results_extra/
```

Important: downloaded Batch API result files are not automatically parsed into PostgreSQL by this command.

### Cached functionality import

```bash
sastllm load_cache_functionalities /path/to/cached_functionalities
```

Loads JSON files named like `functionalities_<snippet_id>.json`, inserts functionality rows, and marks snippets processed.

### Split

```bash
sastllm split
```

Assigns train/test split values to repositories and updates Qdrant point payloads.

Reference: [03_EMBEDDING_AND_SPLITTING.md](./03_EMBEDDING_AND_SPLITTING.md)

### Clustering

```bash
sastllm cluster --mode search
sastllm cluster --mode train
sastllm cluster --mode test
```

Modes:

| Mode | Behavior |
| --- | --- |
| `search` | estimate `k`, save search model and plots |
| `train` | fit MiniBatchKMeans on train embeddings and assign train cluster ids |
| `test` | load trained clusterer and assign test cluster ids |

Reference: [04_FUNCTIONALITY_CLUSTERING.md](./04_FUNCTIONALITY_CLUSTERING.md)

### Classification

```bash
sastllm classify --mode train
sastllm classify --mode test
```

Modes:

| Mode | Behavior |
| --- | --- |
| `train` | train classifier, save checkpoint, write train metrics |
| `test` | load checkpoint, run test, write predictions and metrics |

Reference: [05_REPOSITORY_CLASSIFICATION.md](./05_REPOSITORY_CLASSIFICATION.md)

### Pipeline wrappers

```bash
sastllm train
sastllm test
```

Current wrapper behavior:

```text
train -> cluster --mode train -> classify --mode train
test  -> cluster --mode test  -> classify --mode test
```

These wrappers do not execute loading, functionality generation, embedding creation, or splitting.

## Typical full workflow

```bash
sastllm load
sastllm generate_functionalities
# ensure embeddings exist in Qdrant for functionalities.tag
sastllm split
sastllm cluster --mode train
sastllm classify --mode train
sastllm cluster --mode test
sastllm classify --mode test
```

## Cached functionality workflow

```bash
sastllm load
sastllm load_cache_functionalities /path/to/cached_functionalities
# ensure embeddings exist in Qdrant
sastllm split
sastllm cluster --mode train
sastllm classify --mode train
```

## Model reuse workflow

```bash
sastllm cluster --mode test
sastllm classify --mode test
```

Use this when:

- `configs/clustering.yaml` points to the trained clusterer
- `configs/classification.yaml` points to the trained classifier directory
- test embeddings exist and have Qdrant payload `split=test`

## Output locations

| Location | Content |
| --- | --- |
| `cache/functionalities-<llm_type>/` | parsed functionality caches |
| `api_batches_extra/` | Batch API request JSONL files |
| `batch_results_extra/` | downloaded Batch API outputs/errors |
| `models/clustering/trained_models/` | clustering joblib files |
| `models/classification/trained_models/` | classifier checkpoints, configs, metrics |
| `plots/clustering/searching/` | clustering search plots |
