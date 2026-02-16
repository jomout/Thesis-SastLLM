# Setup

This guide helps you prepare your environment to run SASTLLM locally.

## Requirements

- Linux
- Python 3.12+
- Docker (for Postgres) and docker compose

## 1) Start Postgres with Docker (auto schema)

The repo includes compose files that expose Postgres on port 5433 and automatically load `database/00__init.sql` on first initialization.

```zsh
# from the repo root
# pick a compose file that matches your provider setup
docker compose -f docker-compose-gemini.yml up -d
# or, if you plan to use Anthropic later
# docker compose -f docker-compose-claude.yml up -d
```

## 2) Create a virtual environment and install

```zsh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# Extra dependencies used by clustering and NLP
pip install scikit-learn hdbscan joblib spacy
python -m spacy download en_core_web_sm
```

## 3) Environment variables

Configure Postgres and an LLM provider. You can either export these in your shell or create a `.env` from `.env.example` (the app will read `.env` automatically):

```zsh
export POSTGRES_USER=sastllm
export POSTGRES_PASSWORD=changeme
export POSTGRES_DB=sastllm
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433

# LLM provider credentials (pick one and align with configs/llms.yaml)
export GOOGLE_API_KEY=your_google_api_key     # for Gemini
# export OPENAI_API_KEY=your_openai_api_key   # for OpenAI
```

## 4) Configure YAML files

- `configs/base.yaml`
  - `paths.database_dir`: path to your local dataset root (contains malware/ and benignware/)
  - `paths.evaluation_dir`: directory to store the CodeSearchNet split (optional)
- `configs/llms.yaml`: choose host, model, and params for each processor
- `configs/classification.yaml`: hyperparameters for the ML classifier (train/inference)

## 5) Quick smoke test

```zsh
sastllm --help
```

If the CLI shows commands, your environment is ready. See USAGE.md next.
