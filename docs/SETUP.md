# Setup

This document describes how to install and prepare the **SASTLLM** project for local development and thesis experiments.

The project is packaged as a Python application and exposes a CLI entrypoint:

```bash
sastllm
```

The package name defined in `pyproject.toml` is:

```toml
name = "thesis-sastllm"
```

---

## Requirements

The project currently requires:

- **Python 3.12 or newer**
- a **virtual environment**
- a compatible system for installing the declared Python dependencies
- optional GPU support if the PyTorch CUDA index is used

The Python requirement is defined as:

```toml
requires-python = ">=3.12"
```

---

## Project packaging

The project uses **setuptools** as the build backend.

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
```

The package source directory is:

```toml
[tool.setuptools.package-dir]
"" = "src"
```

The package discovery configuration includes:

- `sastllm*`
- `scripts*`

and excludes:

- `tests*`
- `evaluations*`
- `examples*`

---

## Create a virtual environment

Create and activate a virtual environment before installing the package.

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

---

## Upgrade packaging tools

Upgrade `pip` before installation:

```bash
python -m pip install --upgrade pip
```

---

## Install the project

Install the project in editable mode from the repository root:

```bash
pip install -e .
```

This installs the package and registers the CLI command:

```bash
sastllm
```

The CLI entrypoint is defined as:

```toml
[project.scripts]
sastllm = "scripts.cli:main"
```

---

## Main runtime dependencies

The current `pyproject.toml` declares the following core dependencies:

- `chardet`
- `dotenv`
- `kneed`
- `langchain`
- `langchain-core`
- `langchain-community`
- `langchain-google-genai`
- `langchain-openai`
- `lightning`
- `matplotlib`
- `numpy`
- `psycopg[binary,pool]`
- `qdrant-client`
- `spacy`
- `sqlalchemy`
- `tensorboard`
- `tiktoken`
- `tree-sitter`
- `typer`
- `torch`
- `torchvision`
- `torchaudio`
- `sentence-transformers`
- `datasets`
- `huggingface-hub`
- `shap`
- `ruff`

These support the main pipeline components:

- CLI execution
- LLM integration
- vector embeddings and clustering
- ML training and inference
- NLP preprocessing
- parsing and chunking
- database access
- experiment inspection and visualization

---

## Development dependencies

Optional development dependencies are also defined:

```toml
[project.optional-dependencies]
dev = [
    "ruff>=0.5.0",
    "black>=24.10.0",
    "mypy>=1.10.0",
]
```

Install them with:

```bash
pip install -e ".[dev]"
```

This is recommended if you want linting, formatting, and static type checking in your local workflow.

---

## PyTorch installation notes

The project pins:

- `torch==2.6.0`
- `torchvision==0.21.0`
- `torchaudio==2.6.0`

The `pyproject.toml` also defines a custom `uv` index for CUDA 12.6 wheels:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]
torchvision = [{ index = "pytorch-cu126" }]
torchaudio = [{ index = "pytorch-cu126" }]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

### What this means

- If you use **uv**, the PyTorch packages are intended to resolve from the `pytorch-cu126` index.
- If you use plain `pip`, you may need to install the correct PyTorch build manually depending on your machine, CUDA version, and whether you want CPU-only or GPU-enabled execution.

### Practical advice

If your environment does not support CUDA 12.6, do not blindly force these wheels. Install the PyTorch variant that matches your system.

---

## spaCy model setup

The project depends on `spacy`, but the language model is not installed automatically by the dependency list.

Install the English model explicitly:

```bash
python -m spacy download en_core_web_sm
```

This is commonly needed if your pipeline performs NLP normalization or text preprocessing on functionality descriptions.

---

## Environment variables

The CLI loads environment variables from `.env` on startup through `python-dotenv`.

Create a `.env` file in the project root and define the variables required by your environment.

Typical categories include:

- LLM provider credentials
- database connection settings
- any pipeline-specific runtime secrets or configuration values

### LLM credentials

The project includes both:

- `langchain-google-genai`
- `langchain-openai`

So your `.env` should contain the credentials required by the provider selected in `configs/llms.yaml`.

For example, depending on your provider choice, this usually means defining the corresponding API key.

### Database configuration

The dependency list includes:

- `sqlalchemy`
- `psycopg[binary,pool]`

So make sure your database connection settings are configured if your pipeline uses a PostgreSQL-backed workflow.

---

## Configuration files

Before running the project, review the YAML files under `configs/`.

The current configuration layout includes:

- `configs/base.yaml`
- `configs/llms.yaml`
- `configs/split.yaml`
- `configs/clustering.yaml`
- `configs/classification.yaml`
- `configs/languages.yaml`
- `configs/important_nodes.yaml`
- `configs/comment_nodes.yaml`

At minimum, verify:

- the dataset path in `configs/base.yaml`
- the model/provider in `configs/llms.yaml`
- the train/test configuration in the split, clustering, and classification files

See `docs/CONFIG.md` for a full explanation of each file.

---

## Verify the installation

After installation, test that the CLI is available:

```bash
sastllm --help
```

If everything is installed correctly, Typer should print the available commands.

Typical commands include:

```bash
sastllm load
sastllm split
sastllm generate_functionalities
sastllm cluster --mode train
sastllm classify --mode train
```

---

## Recommended first run

A typical first-run workflow is:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install the project
pip install -e .

# 3. Install the spaCy model
python -m spacy download en_core_web_sm

# 4. Review .env and configs/
# 5. Check CLI availability
sastllm --help
```

Once installation is confirmed, continue with the execution guide in `docs/USAGE.md`.

---

## Linting and formatting

The project includes `ruff` configuration directly in `pyproject.toml`.

Key settings include:

- line length: `150`
- target version: `py312`

Run Ruff with:

```bash
ruff check .
```

Format code with:

```bash
ruff format .
```

If you installed the dev dependencies, you can also use:

```bash
black .
mypy .
```

depending on your preferred workflow.

---

## Editor configuration

The project also defines Pyright virtual environment settings:

```toml
[tool.pyright]
venvPath = "."
venv = ".venv"
```

This is useful if you use:

- VS Code
- Pylance
- Pyright-compatible tooling

Make sure your editor is pointing to the `.venv` environment in the project root.

---

## Troubleshooting checklist

If setup fails, check the following first:

- Are you using **Python 3.12+**?
- Is the virtual environment activated?
- Did `pip install -e .` complete successfully?
- Did you install `en_core_web_sm` for spaCy?
- Are your `.env` variables present?
- Does `configs/base.yaml` point to a valid dataset path?
- Does your selected LLM provider have valid credentials?
- Is your PyTorch installation compatible with your machine?

---

## Minimal install summary

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python -m spacy download en_core_web_sm
sastllm --help
```

---

## Related documentation

After setup, continue with:

1. `docs/CONFIG.md`
2. `docs/USAGE.md`
3. `docs/PIPELINE.md`

That order is the cleanest path from installation to execution to understanding the system.
