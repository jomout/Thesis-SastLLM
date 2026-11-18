# Setup

This document describes local setup for the SAST-LLM thesis project.

## Requirements

The package currently declares:

```toml
requires-python = ">=3.14"
```

Recommended local tools:

- Python 3.14 or newer
- `uv` or `pip`
- Docker and Docker Compose for PostgreSQL and Qdrant
- API credentials for the configured LLM provider

## Install

Create and activate a virtual environment:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
python -m pip install --upgrade pip
pip install -e .
```

With development tools:

```bash
pip install -e ".[dev]"
```

The CLI entrypoint is:

```bash
argus
```

It is defined in `pyproject.toml` as:

```toml
[project.scripts]
argus = "scripts.cli:main"
```

## Runtime services

Use `docker-compose.argus.yml` for the ARGUS services. Include `-f docker-compose.argus.yml` in Compose commands so they target the ARGUS stack even if you have a local Compose file.

Start PostgreSQL and Qdrant:

```bash
docker compose -f docker-compose.argus.yml up -d
uv run alembic upgrade head
```

Services:

| Service            | Purpose                     |
| ------------------ | --------------------------- |
| `argus_database`   | PostgreSQL relational store |
| `argus_embeddings` | Qdrant vector store         |

Docker Compose creates the PostgreSQL service and persistent volume. Alembic then creates or upgrades the schema. The SQL files under `database/` are retained as legacy bootstrap and reference artifacts, but Compose does not apply them automatically:

- `database/00__init.sql`
- `database/01__triggers.sql`

## Database migrations

Alembic migrations under `database/migrations/` are the authoritative schema history. Configuration comes from the same `POSTGRES_*` environment variables as the application.

Create or upgrade a fresh database:

```bash
docker compose -f docker-compose.argus.yml up -d argus_database
uv run alembic upgrade head
uv run alembic current
```

Review generated SQL without connecting to PostgreSQL:

```bash
uv run alembic upgrade head --sql
```

When changing ORM metadata, generate a candidate revision and review it before applying it. PostgreSQL functions and triggers must be written explicitly because Alembic autogeneration does not discover them reliably:

```bash
uv run alembic revision --autogenerate -m "describe schema change"
uv run alembic check
uv run alembic upgrade head
```

### Adopt an existing database

Databases originally created by `database/00__init.sql` and `database/01__triggers.sql` already match revision `0001_core_schema`. Back up the database and verify that it contains the four core tables, their indexes, and the processed-state triggers before stamping it. Stamping records a revision without executing its schema operations.

```bash
uv run alembic stamp 0001_core_schema
uv run alembic upgrade head
uv run alembic current
```

The upgrade adds `csn_snippets`, `repository_predictions`, and the `alembic_version` table without rebuilding or deleting the populated core tables. Do not stamp a database whose schema has drifted from the legacy SQL; reconcile that drift in a dedicated migration first.

Downgrades can remove tables and data. Use them only after reviewing the target revision and taking a verified backup; never use `downgrade base` on experiment state that must be preserved.

## Environment variables

Create `.env` in the project root.

Common values:

```text
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ENDPOINT_URL=http://localhost:8000
ACCESS_TOKEN=access_key
```

Use the API key required by the host in `configs/llms.yaml`.

## Dependencies

Major runtime dependency groups:

- Typer for the CLI
- SQLAlchemy and psycopg for PostgreSQL
- Alembic for PostgreSQL schema migrations
- Qdrant client for vector storage
- Tree-sitter and tiktoken for parsing/chunking
- LangChain Google/OpenAI integrations for LLM calls
- Sentence Transformers for embeddings
- scikit-learn through project dependencies for splitting and clustering usage
- Lightning and PyTorch for repository classification
- SHAP and Matplotlib for analysis/plots

## PyTorch index

`pyproject.toml` currently points `uv` PyTorch packages to the CPU wheel index:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

If you need GPU training, install a PyTorch build that matches your CUDA/runtime environment.

## Verify

Check CLI availability:

```bash
argus --help
```

Check services:

```bash
docker compose -f docker-compose.argus.yml ps
```

Then configure the dataset path in `configs/base.yaml` and follow [USAGE.md](./USAGE.md).
