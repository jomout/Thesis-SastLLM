# SASTLLM

LLM-powered static analysis for malware detection. This project ingests source repositories (malware and benignware), chunks files into code snippets, uses an LLM to describe snippet functionality, clusters those functionalities, and classifies repositories using an ML model.

Use this README for a quick overview and getting started. For deeper topics, see the docs:

- docs/SETUP.md – Environment, installation, and Postgres
- docs/USAGE.md – CLI commands and typical workflows
- docs/CONFIG.md – YAML configuration (base, llms, classification)
- docs/PIPELINE.md – End-to-end pipeline, prompts, and processors
- docs/DB_SCHEMA.md – Database models and relationships
- docs/TROUBLESHOOTING.md – Common issues and fixes

## Highlights

- Multi-language chunking via Tree-sitter with token-budget aware segmentation
- Snippet functionality extraction via LLM with AST-augmented prompts
- Vector embeddings, PCA, and clustering for functionality tags
- Classification:
  - ML: Per-repo cluster histograms encoded to features → MLP
- Postgres-backed storage for repositories, files, snippets, functionalities, clusters

## Architecture (high level)

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

## Quickstart

Prerequisites:

- Linux, Python 3.12+
- Docker (for Postgres)

1. Start Postgres (port 5433 on host → 5432 in container)

```zsh
# Choose one of the provided compose files:
# - docker-compose-gemini.yml (default in docs) uses Gemini provider
# - docker-compose-claude.yml uses Anthropic (if you switch providers later)
docker compose -f docker-compose-gemini.yml up -d
```

1. Create a virtual environment and install the package (editable) and required extras:

```zsh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# Extra deps used by clustering and NLP
pip install scikit-learn hdbscan joblib spacy
python -m spacy download en_core_web_sm
```

1. Set environment variables (adjust to your setup):

```zsh
export POSTGRES_USER=sastllm
export POSTGRES_PASSWORD=changeme
export POSTGRES_DB=sastllm
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433

# LLM provider credentials (pick one that matches configs/llms.yaml)
export GOOGLE_API_KEY=your_google_api_key     # for Gemini
# export OPENAI_API_KEY=your_openai_api_key   # for OpenAI
```

1. Configure paths in `configs/base.yaml`:

- `paths.database_dir`: root folder with your repositories (see below)
- `paths.evaluation_dir`: where CodeSearchNet split will be downloaded (optional)

Expected dataset layout for `paths.database_dir`:

```text
/path/to/dataset/
  malware/
    repo1/
      ...
  benignware/
    repoA/
      ...
```

1. Run the pipeline via CLI:

```zsh
# Insert repositories/files/snippets into DB from local dataset
sastllm setup

# Generate snippet functionalities (LLM) and cluster tags
sastllm train

# ML-based classification (see docs/USAGE.md)
sastllm classify_1

# Optional: evaluation with CodeSearchNet snippets/docstrings
sastllm eval
```

See docs/USAGE.md for detailed command descriptions and options.

## Configuration files

- `configs/base.yaml` – app name, logging, and paths for datasets/evaluation
- `configs/llms.yaml` – model host/name/params for each processor stage
- `configs/classification.yaml` – training/inference hyperparameters for the MLP route

Details and examples in docs/CONFIG.md.

## Project layout

```text
src/
  sastllm/
    analyzers/      # LLM prompt pipelines (LangChain chains)
    extractors/     # Regex/structured text extractors for LLM outputs
    parsers/        # AST/Tree-sitter parsing and chunking
    processors/     # Orchestrators for each pipeline step
    utils/          # Normalizer, embeddings, clustering, repository encoder/classifier
    models/         # PyTorch Lightning datasets and models
scripts/
  cli.py           # Typer CLI entry
  pipelines.py     # CLI action implementations
  evaluation.py    # CSN-based evaluation helpers
  logging_config.py
configs/
  base.yaml, llms.yaml, classification.yaml
```

## Notes and caveats

- LLM providers: default config uses Google Gemini 2.5 Flash via LangChain for intermediate analysis (e.g., snippet functionalities). You can switch to OpenAI by changing `host: openai` and setting `OPENAI_API_KEY`.
- Docker compose files: this repo ships `docker-compose-gemini.yml` (default in docs) and `docker-compose-claude.yml`. Use `-f <file>` with `docker compose`.
- Clustering: Requires `scikit-learn` and `hdbscan`. Ensure both are installed.
- spaCy: `en_core_web_sm` model must be installed for normalization.
- Postgres: the compose files expose port 5433 on the host; set matching `POSTGRES_PORT`.
  - The schema file `database/00__init.sql` is mounted into the container's `/docker-entrypoint-initdb.d/` and will be executed automatically on first initialization of the volume. To re-run it, remove the `pg_sastllm_data` volume.
  - The app loads environment variables from `.env` via `python-dotenv`. Set `DB_SKIP_CREATE_ALL=true` if you want to rely solely on the SQL schema and avoid SQLAlchemy `create_all`.
- Dataset labeling: The ML route expects correct repository labels (malware/benignware). See docs/PIPELINE.md for labeling behavior and recommendations.

## Next steps and contributions

Improvements that add value quickly:

- Persist LLM classification outputs back to DB
- Refine repository label inference during dataset load
- Finalize and document the ML classifier training/inference path end-to-end
- Expand tests and add linters/formatters

PRs and issues are welcome.

