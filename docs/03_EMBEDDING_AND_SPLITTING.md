# Embedding And Dataset Splitting

This stage manages repository train/test assignment and Qdrant payload metadata for functionality embeddings.

## CLI entrypoints

```bash
argus embed
argus split
```

Implementation paths:

- Pipeline wrapper: `src/scripts/pipelines.py::split_dataset`
- Splitter: `src/argus/utils/dataset_splitter.py::DatasetSplitter`
- Embedder: `src/argus/cluster/embedder.py::Embedder`
- Qdrant access: `src/argus/db/managers/embeddings_manager.py`

## Current command behavior

`argus embed` reads `configs/split.yaml`, constructs `DatasetSplitter`, and calls `embed_all_repositories()`. Existing Qdrant point ids are skipped.

The `split_dataset()` implementation constructs the same service and calls:

```python
database_splitter.split_repositories(train_size=train_size, test_size=test_size)
```

Embedding remains a separate explicit operation because it is expensive and stateful. `argus split` updates split labels in PostgreSQL and Qdrant payloads, and assumes `argus embed` has already populated the collection. The SentenceTransformer is initialized lazily by the embedding operation, so `argus split` does not load or contact the model.

## Split configuration

Current config:

```yaml
split:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  training:
    ratio: 0.7
  testing:
    ratio: 0.3
  binary_labels:
    - 0: "benign"
    - 1: "malicious"
```

The Qdrant collection name is derived from the model name by replacing `/` with `_`:

```text
sentence-transformers_all-mpnet-base-v2
```

## Splitting algorithm

The splitter:

1. Fetches all repositories from PostgreSQL.
2. Builds pairs of `(repository_id, label)`.
3. Asserts that `train_size + test_size == 1.0`.
4. Uses `sklearn.model_selection.train_test_split`.
5. Stratifies by repository label.
6. Uses random seed `42`.
7. Writes `repositories.split` in bulk.
8. Updates Qdrant payload field `split` for points whose `repository_id` is in each split.

If `test_size` is exactly `1.0`, all repositories go to test and train is empty.

## Embedding helper behavior

`argus embed` invokes `DatasetSplitter.embed_all_repositories()`, which can:

1. Fetch all repositories.
2. Fetch all functionalities grouped by repository.
3. Flatten `(functionality_id, tag, repository_id)` tuples.
4. Embed tags in batches of `256`.
5. Insert vectors into Qdrant with payload:

```json
{
  "repository_id": 123,
  "split": "full",
  "tag": "normalized functionality tag"
}
```

The embedder uses `SentenceTransformer` and checks existing Qdrant ids to avoid recomputing cached vectors. New embeddings produced by `Embedder.embed()` are returned to the caller, and `DatasetSplitter.embed_all_repositories()` persists them through `EmbeddingsManager.insert_embeddings()`.

## Qdrant storage model

Each functionality embedding is stored as one Qdrant point:

| Qdrant field            | Meaning                                |
| ----------------------- | -------------------------------------- |
| point id                | `functionality_id`                     |
| vector                  | embedding of `functionalities.tag`     |
| `repository_id` payload | repository that owns the functionality |
| `split` payload         | `full`, `train`, or `test`             |
| `tag` payload           | normalized functionality tag           |

Distance is cosine similarity.

## Dependencies between stages

Splitting is a bridge between functionality generation and clustering:

```text
functionalities.tag
  -> sentence-transformer embedding
  -> Qdrant point payload split
  -> cluster train/test source
```

Clustering expects Qdrant points to be available and marked with the proper `split` payload.

## Operational notes

- If Qdrant has no collection for the configured embedding model, split payload updates will fail.
- The splitter updates database splits and Qdrant splits separately; keep both stores in sync.
- Label strings are preserved during splitting. Later classification maps every non-`benign` label to `malicious`.
- The code supports only train/test split in this stage, not a persisted validation split.
