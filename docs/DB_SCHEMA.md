# Database schema

The project uses SQLAlchemy ORM with a Postgres backend. Tables are created automatically on first use.

## Entities

- `repositories`
  - `repository_id` (PK)
  - `name` (unique)
  - `label` (nullable): ground-truth class e.g., `malware` or `benignware`

- `files`
  - `file_id` (PK)
  - `repository_id` (FK → repositories)
  - `language` (e.g., `py`, `cpp`)
  - `filename`
  - `filepath` (path relative to dataset root)

- `snippets`
  - `snippet_id` (PK)
  - `file_id` (FK → files)
  - `start_line`, `end_line`
  - `code` (text)

- `functionalities`
  - `functionality_id` (PK)
  - `snippet_id` (FK → snippets)
  - `description` (raw LLM output sentence)
  - `tag` (normalized description)
  - `cluster_id` (nullable FK → clusters)

- `clusters`
  - `cluster_id` (PK)
  - `label` (nullable; LLM pseudolabel)

- `csn_snippets` (evaluation)
  - `csn_snippet_id` (PK)
  - `repository`, `filepath`
  - `start_line`, `end_line`
  - `code`
  - `functionality` (CSN docstring)

## Useful streaming queries

- `get_snippets_with_file_meta()` → for snippet LLM stage
- `get_functionalities()` → to fetch all tags for clustering
- `get_clusters_with_tags()` → for cluster pseudolabel stage
- `get_files_with_labels()` → for file-level flagging
- `get_repositories_with_labels()` → for repository ground-truth labels
- `get_repositories_with_cluster_ids()` → for ML classification features

## Suggested prediction tables (future)

Add persistent results for LLM/ML classification:

- `repository_predictions`
  - `id` (PK)
  - `repository_id` (FK → repositories)
  - `classification` (text: `malware`/`benignware`)
  - `probabilities_json` (jsonb; optional per-class probabilities or scores)
  - `created_at` (timestamp)

- `file_flags`
  - `id` (PK)
  - `file_id` (FK → files)
  - `flags_json` (jsonb)
  - `justification` (text)
  - `created_at` (timestamp)
