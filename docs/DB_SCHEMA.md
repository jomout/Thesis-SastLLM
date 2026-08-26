# Database Schema

This document describes the PostgreSQL schema used by the SAST-LLM pipeline. For the broader storage picture, including Qdrant, see [06_STORAGE_AND_DATA_ACCESS.md](./06_STORAGE_AND_DATA_ACCESS.md).

## Core hierarchy

```text
repositories -> files -> snippets -> functionalities
```

Each level represents a finer unit of analysis:

- repository: one analyzed project
- file: one source file in that repository
- snippet: one chunked code segment
- functionality: one LLM-derived semantic action for a snippet

## Global conventions

- Main tables include `created_at` and `updated_at`.
- `processed` flags track pipeline progress on repositories, files, and snippets.
- Child rows cascade when a parent repository/file/snippet is deleted.
- `updated_at` is maintained by trigger functions in SQL and/or SQLAlchemy `onupdate` where configured.

## `repositories`

Fields:

| Field           | Meaning                                              |
| --------------- | ---------------------------------------------------- |
| `repository_id` | primary key                                          |
| `name`          | unique repository identifier                         |
| `label`         | source label, such as `benign`, `apt`, `rat`, `worm` |
| `processed`     | true when all child files/snippets are processed     |
| `split`         | `train` or `test`                                    |
| `created_at`    | insert timestamp                                     |
| `updated_at`    | update timestamp                                     |

Used by loading, splitting, processed filtering, and classification.

## `files`

Fields:

| Field           | Meaning                                      |
| --------------- | -------------------------------------------- |
| `file_id`       | primary key                                  |
| `repository_id` | parent repository                            |
| `language`      | internal language name from `languages.yaml` |
| `filename`      | basename                                     |
| `filepath`      | path relative to dataset root                |
| `processed`     | true when all child snippets are processed   |
| `created_at`    | insert timestamp                             |
| `updated_at`    | update timestamp                             |

Important indexes:

- `idx_files_repository_id`
- `idx_files_repo_processed`

## `snippets`

Fields:

| Field        | Meaning                                    |
| ------------ | ------------------------------------------ |
| `snippet_id` | primary key                                |
| `file_id`    | parent file                                |
| `start_line` | 1-based start line                         |
| `end_line`   | 1-based inclusive end line                 |
| `code`       | chunked source code                        |
| `processed`  | true after functionality generation/import |
| `created_at` | insert timestamp                           |
| `updated_at` | update timestamp                           |

Important indexes:

- `idx_snippets_file_id`
- `idx_snippets_processed`
- `idx_snippets_file_id_processed`

## `functionalities`

Fields:

| Field              | Meaning                           |
| ------------------ | --------------------------------- |
| `functionality_id` | primary key                       |
| `snippet_id`       | parent snippet                    |
| `description`      | raw LLM-generated action sentence |
| `tag`              | normalized functionality text     |
| `cluster_id`       | assigned semantic cluster id      |
| `created_at`       | insert timestamp                  |
| `updated_at`       | update timestamp                  |

Important indexes:

- `idx_functionalities_snippet_id`
- `idx_functionalities_cluster_id`
- `idx_functionalities_tag`

## Additional ORM models

The ORM also defines:

| Model/table                                            | Purpose                                     |
| ------------------------------------------------------ | ------------------------------------------- |
| `CSNCodeSnippetModel` / `csn_snippets`                 | CodeSearchNet snippet/functionality storage |
| `RepositoryPredictionModel` / `repository_predictions` | optional repository prediction records      |

The current classifier writes prediction and metric JSON files under model directories rather than inserting into `repository_predictions`.

## Processed propagation triggers

`database/01__triggers.sql` defines:

- `recompute_file_processed(file_id)`
- `recompute_repository_processed(repository_id)`
- `trg_snippet_refresh_file_processed()`
- `trg_file_refresh_repository_processed()`

Trigger propagation:

```text
snippets.processed changes
  -> recompute files.processed
  -> recompute repositories.processed
```

A file is processed when no child snippet has `processed IS NOT TRUE`.

A repository is processed when no child file has `processed IS NOT TRUE`.

This is important because classification reads only repositories where:

```sql
repositories.processed IS TRUE
```

## Classification aggregation

The classifier does not read raw snippets directly. It aggregates cluster ids through:

```text
repositories
  left join files
  left join snippets
  left join functionalities
```

For each processed repository it builds:

```text
{cluster_id: count}
```

That dictionary becomes the repository-level feature vector.

## Schema/data flow

```text
argus load
  -> repositories/files/snippets

argus generate_functionalities
  -> functionalities
  -> snippets.processed=true
  -> trigger updates files/repositories processed

argus split
  -> repositories.split

argus cluster --mode ...
  -> functionalities.cluster_id

argus classify --mode ...
  -> reads processed repositories and cluster counts
```
