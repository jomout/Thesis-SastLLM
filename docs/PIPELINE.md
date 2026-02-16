# Pipeline

This document describes the end-to-end flow, key modules, and contracts.

## Overview

```text
Local Dataset (malware/, benignware/)
        |
        v
CodeProcessor (parsers/chunker)  -> DB: repositories, files, snippets
        |
        v
SnippetProcessor (LLM prompt -> extract -> normalize)
        -> DB: functionalities(description, tag)
        |
        v
TagProcessor (embeddings -> PCA -> HDBSCAN+KMeans)
        -> DB: clusters + functionality.cluster_id
        |
        +--> ClusterProcessor (LLM pseudolabels for clusters) -> DB: clusters.label
        |
        v
Classification
  - ML: RepositoryEncoder -> MLP (Lightning)
```

## Ingestion and chunking

- `CodeProcessor` walks `paths.database_dir` recursively collecting files (optionally filtered by suffix).
- Uses `CodeChunker` which:
  - Builds a Tree-sitter AST for the file (via `ASTParser`).
  - Finds “breakpoints” (functions, classes) and comment lines per language.
  - Packs lines into chunks under a token budget (tiktoken).
- Inserts `repositories`, `files`, and `snippets` into Postgres.
- Labeling: derive `RepositoryModel.label` from the path (e.g., top-level folder `malware/` vs `benignware/`). If not present, you can adapt the loader accordingly.

## Snippet functionality extraction (LLM)

- `FunctionalityPromptGenerator`:
  - Parses snippet code with Tree-sitter to extract:
    - function calls, control structures, string literals
  - Formats a structured snippet payload (file path, language, AST lists, code block).
- `FunctionalityAnalyzer` (LangChain ChatPrompt):
  - System prompt: expert malware detection scanner
  - User prompt: strict output format per chunk, concise sentence list
- `FunctionalityExtractor` parses the Chat response to a structure:

```python
{
  "Chunk 123": {
    "functionalities": [
      "Connects to a remote server.",
      "Logs keystrokes.",
    ]
  }
}
```

- `Normalizer` (spaCy) lemmatizes and filters each sentence into a tag string.
- Persist to `functionalities(description, tag, cluster_id=NULL)`.

## Clustering and pseudolabels

- `TagProcessor` pulls all functionality tags.
- `Clusterer` steps:
  - SentenceTransformer embeddings
  - PCA (retain ~90% variance) + L2 normalize
  - HDBSCAN to estimate cluster count
  - KMeans for final labels
- Persist:
  - `clusters` rows (if not present) and set `functionalities.cluster_id`.
- `ClusterProcessor` (optional): generate pseudolabels via LLM per cluster using tags as context, write to `clusters.label`.

## Classification (ML)

- `RepositoryClassifier`:
  - `get_repositories_with_cluster_ids()` returns per-repo cluster counts and the ground-truth label
  - `RepositoryEncoder` builds a per-repo percentage vector of clusters
  - Lightning MLP (`models/model.py`) + DataModule (`models/dataset.py`)
  - See `models/classify.py` and `RepositoryClassifier.run()` for training/inference.

## Prompts (summary)

- Functionality: concise sentences per chunk. See `analyzers/functionality_analyzer.py`.
- Cluster pseudolabel: short label + justification. See `analyzers/cluster_analyzer.py`.
  

## Error handling & batching

- All processors iterate in batches; DB queries use streaming (`yield_per`).
- LLM stages include sleep intervals between batches to avoid rate limits.
- Exceptions in batch loops are logged and processing continues.

## Performance notes

- Embeddings (SentenceTransformers) can leverage GPU if available.
- For large datasets, consider larger chunk token budgets and batch sizes that match your model limits.
- Indexing in Postgres on foreign keys can speed up joins.
