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

Start PostgreSQL and Qdrant:

```bash
docker compose up -d
```

Services:

| Service           | Purpose                     |
| ----------------- | --------------------------- |
| `code_database`   | PostgreSQL relational store |
| `code_embeddings` | Qdrant vector store         |

The current Compose file mounts a database backup SQL file on first initialization. The schema files under `database/` are also available as reference:

- `database/00__init.sql`
- `database/01__triggers.sql`

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
docker compose ps
```

Then configure the dataset path in `configs/base.yaml` and follow [USAGE.md](./USAGE.md).
