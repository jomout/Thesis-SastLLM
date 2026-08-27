# Storage And Data Access

SAST-LLM uses PostgreSQL for relational pipeline state and Qdrant for functionality embeddings.

## Services

`docker-compose.yml` defines:

| Service           | Image                 | Ports                   | Purpose                   |
| ----------------- | --------------------- | ----------------------- | ------------------------- |
| `code_database`   | `postgres:18-alpine`  | `${POSTGRES_PORT}:5432` | relational pipeline state |
| `code_embeddings` | `qdrant/qdrant:v1.17` | `6333`, `6334`          | vector embeddings         |

PostgreSQL environment variables:

```text
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_DB
POSTGRES_PORT
```

Python database connection settings are read from `.env` through `pydantic-settings`:

```text
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_DB
POSTGRES_HOST
POSTGRES_PORT
```

## PostgreSQL ORM models

Main models live in `src/argus/db/models.py`.

### `repositories`

Top-level repository record.

Fields:

- `repository_id`
- `name`
- `label`
- `processed`
- `split`
- `created_at`
- `updated_at`

Used by:

- dataset loading
- train/test splitting
- processed filtering
- classification labels

### `files`

Source files belonging to a repository.

Fields:

- `file_id`
- `repository_id`
- `language`
- `filename`
- `filepath`
- `processed`
- `created_at`
- `updated_at`

### `snippets`

Chunked code snippets belonging to a file.

Fields:

- `snippet_id`
- `file_id`
- `start_line`
- `end_line`
- `code`
- `processed`
- `created_at`
- `updated_at`

### `functionalities`

LLM-derived semantic functionality descriptions.

Fields:

- `functionality_id`
- `snippet_id`
- `description`
- `tag`
- `cluster_id`
- `created_at`
- `updated_at`

### `csn_snippets`

CodeSearchNet snippet storage model.

Fields:

- `csn_snippet_id`
- `repository`
- `filepath`
- `start_line`
- `end_line`
- `code`
- `functionality`
- `created_at`
- `updated_at`

This model exists in the ORM, although it is not part of the main repository -> file -> snippet -> functionality chain.

### `repository_predictions`

Prediction storage model for repository-level outputs.

Fields:

- `id`
- `repository_id`
- `method`
- `classification`
- `flags_json`
- `justification`
- `created_at`
- `updated_at`

The current classifier writes metrics and predictions to JSON files under model directories rather than inserting rows here.

## Processed propagation

`database/01__triggers.sql` defines trigger functions that propagate processed status:

```text
snippets.processed -> files.processed -> repositories.processed
```

Behavior:

- A file is processed when it has no unprocessed snippets.
- A repository is processed when it has no unprocessed files.
- Triggers run after snippet and file inserts, updates, and deletes.
- A backfill section recomputes existing rows and is safe to rerun.

This matters because classification filters to `repositories.processed = true`.

## Qdrant collections

Embeddings are stored in Qdrant collections named after the sentence-transformer model:

```text
sentence-transformers/all-mpnet-base-v2
  -> sentence-transformers_all-mpnet-base-v2
```

Each point uses:

| Qdrant item           | Value                          |
| --------------------- | ------------------------------ |
| id                    | `functionality_id`             |
| vector                | sentence-transformer embedding |
| distance              | cosine                         |
| payload.repository_id | owning repository              |
| payload.split         | `full`, `train`, or `test`     |
| payload.tag           | normalized functionality tag   |

## Data managers

Managers under `src/argus/db/managers/` provide typed access around SQLAlchemy and Qdrant:

| Manager                | Main responsibility                                     |
| ---------------------- | ------------------------------------------------------- |
| `RepositoryManager`    | repositories, split updates, classification aggregation |
| `FileManager`          | file rows                                               |
| `SnippetManager`       | snippet rows and unprocessed snippet iteration          |
| `FunctionalityManager` | functionality insert/update/query                       |
| `EmbeddingsManager`    | Qdrant insert, retrieval, count, payload update         |

Most SQL writes use transaction contexts. Bulk operations are used for snippets, functionalities, split updates, and cluster-id assignment.

## Data lineage summary

```text
dataset path
  -> repositories/files/snippets in PostgreSQL
  -> functionalities in PostgreSQL
  -> functionality embeddings in Qdrant
  -> train/test split in PostgreSQL and Qdrant payloads
  -> cluster ids in PostgreSQL
  -> repository vectors in memory
  -> model artifacts and metric JSON files
```
