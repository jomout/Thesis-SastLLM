# Pipeline

This document describes the end-to-end SAST-LLM thesis pipeline and links each stage to its detailed reference file.

## Overview

SAST-LLM transforms raw source repositories into semantic functionality representations and classifies repositories from their behavioral cluster profile.

```text
Code repositories
  -> data ingestion and chunking
  -> snippet functionality generation
  -> functionality embedding and dataset split metadata
  -> functionality clustering
  -> repository vectorization
  -> repository classification
```

## Pipeline diagram

![Updated SAST-LLM Pipeline](./pipeline_diagram.png)

## Stage map

| Stage | Main command | Main output | Detailed reference |
| --- | --- | --- | --- |
| 1. Data ingestion and chunking | `sastllm load` | `repositories`, `files`, `snippets` | [01_DATA_INGESTION_AND_CHUNKING.md](./01_DATA_INGESTION_AND_CHUNKING.md) |
| 2. Functionality generation | `sastllm generate_functionalities` | `functionalities.description`, `functionalities.tag` | [02_FUNCTIONALITY_GENERATION.md](./02_FUNCTIONALITY_GENERATION.md) |
| 3. Embedding and splitting | `sastllm split` | `repositories.split`, Qdrant `split` payloads | [03_EMBEDDING_AND_SPLITTING.md](./03_EMBEDDING_AND_SPLITTING.md) |
| 4. Functionality clustering | `sastllm cluster --mode ...` | `functionalities.cluster_id`, clustering model | [04_FUNCTIONALITY_CLUSTERING.md](./04_FUNCTIONALITY_CLUSTERING.md) |
| 5. Repository classification | `sastllm classify --mode ...` | model artifacts, predictions, metrics | [05_REPOSITORY_CLASSIFICATION.md](./05_REPOSITORY_CLASSIFICATION.md) |

Storage and configuration references:

- [06_STORAGE_AND_DATA_ACCESS.md](./06_STORAGE_AND_DATA_ACCESS.md)
- [07_CONFIGURATION_REFERENCE.md](./07_CONFIGURATION_REFERENCE.md)

## End-to-end data flow

```text
Dataset directory
  -> CodeProcessor
  -> PostgreSQL repositories/files/snippets
  -> SnippetProcessor or cached functionality loader
  -> PostgreSQL functionalities
  -> SentenceTransformer embeddings
  -> Qdrant vectors
  -> train/test split payloads
  -> MiniBatchKMeans cluster ids
  -> repository cluster-count vectors
  -> Lightning binary classifier
  -> prediction and metric JSON files
```

## Practical execution order

A full experiment from raw local dataset usually follows:

```bash
sastllm load
sastllm generate_functionalities
# ensure functionality embeddings exist in Qdrant
sastllm split
sastllm cluster --mode train
sastllm classify --mode train
sastllm cluster --mode test
sastllm classify --mode test
```

The compact wrappers run only the final model stages:

```bash
sastllm train
sastllm test
```

Current wrapper behavior:

```text
train -> cluster --mode train -> classify --mode train
test  -> cluster --mode test  -> classify --mode test
```

They do not run dataset loading, LLM generation, splitting, or embedding creation.

## Current implementation caveats

- `sastllm split` currently calls `split_repositories()` only. The available `embed_all_repositories()` helper is present but commented out in `src/scripts/pipelines.py`.
- Classification filters repositories with `processed=true`, so trigger propagation from snippets to files to repositories matters.
- Classification treats every label other than exact `benign` as `malicious`.
- The classification config accepts `l1_param` as an alias for the runtime `l1_lambda` field.
- Repository classification assumes repository ids behave like dense, 1-based indices when building data module indices.

## Thesis interpretation

The classifier does not operate directly on raw source code. Instead, the project builds a semantic abstraction:

```text
source code -> snippets -> functionality tags -> clusters -> repository vector
```

This makes the decision surface easier to inspect because each repository prediction is based on the distribution of clustered functionality descriptions.
