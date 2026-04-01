# Database Schema

This project uses a **PostgreSQL** database to store and process repository-level intelligence.
The schema is designed as a hierarchical pipeline:

    ```
    repositories → files → snippets → functionalities
    ```

Each level represents a finer granularity of code analysis.

---

## Global Conventions

* All tables include:

  * `created_at`: timestamp of insertion
  * `updated_at`: auto-updated timestamp via trigger
* `processed` flags are used to track pipeline progress
* Cascading deletes ensure referential integrity

---

## Trigger: `set_updated_at`

Automatically updates `updated_at` on row modification.

* Applied via `BEFORE UPDATE` triggers on all tables
* Uses `clock_timestamp()` for precise timing

---

## Entities

### `repositories`

Represents a source code repository (e.g., a GitHub repo).

**Fields:**

* `repository_id` (PK)
* `name` (unique, required): repository identifier
* `label` (nullable): ground-truth classification (e.g., `malware`, `benignware`)
* `processed` (boolean): whether repository-level processing is complete
* `split` (nullable): dataset split (`train`, `val`, `test`)
* `created_at`, `updated_at`

**Notes:**

* Top-level entity in the pipeline
* Used for ML dataset partitioning and evaluation

---

### `files`

Represents individual files within a repository.

**Fields:**

* `file_id` (PK)
* `repository_id` (FK → repositories, cascade delete)
* `language` (required): programming language (e.g., `py`, `cpp`)
* `filename`
* `filepath`: relative path inside repository
* `processed` (boolean): whether file has been analyzed
* `created_at`, `updated_at`

**Indexes:**

* `idx_files_repository_id`
* `idx_files_repo_processed (repository_id, processed)`

**Notes:**

* Enables filtering by language and processing status
* Core unit for file-level analysis

---

### `snippets`

Represents extracted code segments from files.

**Fields:**

* `snippet_id` (PK)
* `file_id` (FK → files, cascade delete)
* `start_line`, `end_line`: location in file
* `code`: raw code snippet
* `processed` (boolean): whether snippet has been analyzed by LLM
* `created_at`, `updated_at`

**Indexes:**

* `idx_snippets_file_id`
* `idx_snippets_processed`
* `idx_snippets_file_id_processed (file_id, processed)`

**Notes:**

* Main unit for LLM-based code understanding
* Supports incremental processing via `processed` flag

---

### `functionalities`

Represents semantic descriptions extracted from snippets.

**Fields:**

* `functionality_id` (PK)
* `snippet_id` (FK → snippets, cascade delete)
* `description`: raw natural language output (e.g., LLM-generated)
* `tag`: normalized/cleaned functionality label
* `cluster_id` (nullable): grouping identifier for similar functionalities
* `created_at`, `updated_at`

**Indexes:**

* `idx_functionalities_snippet_id`
* `idx_functionalities_cluster_id`
* `idx_functionalities_tag`

**Notes:**

* Bridges code → semantics
* `tag` is used for downstream clustering and ML features
* `cluster_id` is optional and can be assigned post-processing

---

## Data Flow

1. **Repositories ingested**
2. Files extracted and stored in `files`
3. Files split into `snippets`
4. Snippets analyzed → `functionalities`
5. Functionalities optionally clustered via `cluster_id`

---

## Design Highlights

* **Pipeline-friendly:** `processed` flags allow resumable workflows
* **Efficient queries:** composite indexes support fast filtering
* **Cascade deletes:** deleting a repository removes all dependent data
* **Extensible:** `functionalities.cluster_id` enables future clustering without schema changes

---

## Future Extensions (Suggested)

### `clusters` (optional)

If clustering becomes a first-class concept:

* `cluster_id` (PK)
* `label` (optional human-readable label)

---

### Prediction Tables

#### `repository_predictions`

* `id` (PK)
* `repository_id` (FK)
* `classification`
* `probabilities_json` (jsonb)
* `created_at`

#### `file_flags`

* `id` (PK)
* `file_id` (FK)
* `flags_json` (jsonb)
* `justification`
* `created_at`

---

## Summary

The schema is designed for **multi-stage code intelligence processing**, enabling:

* Fine-grained code analysis (snippet level)
* Semantic understanding (functionality extraction)
* Scalable ML pipelines (via tags and clusters)
* Efficient incremental processing
