# ARGUS

**ARGUS** (**A**utomated **R**ecognition and **G**uarding against **U**ntrusted **S**ource code) is an LLM-assisted static malware analysis framework. It transforms source code repositories into **semantic functionality representations** and classifies them based on their behavioral profiles.

At a high level, the framework:

1. ingests source repositories
2. splits source files into code snippets
3. generates functionality descriptions for each snippet with an LLM
4. clusters semantically similar functionalities
5. builds repository-level vectors from functionality clusters
6. performs repository classification

This README provides the central overview of the project. More detailed documentation is available in the `docs/` directory.

## Documentation

- `docs/SETUP.md` – environment setup, installation, and dependencies
- `docs/USAGE.md` – CLI commands and execution workflows
- `docs/CONFIG.md` – YAML configuration files and their roles
- `docs/PIPELINE.md` – end-to-end pipeline and phase descriptions
- `docs/DB_SCHEMA.md` – database schema and model relationships
- `docs/README.md` – documentation index and stage-specific reference files

---

## Core idea

Instead of classifying repositories directly from raw source code, ARGUS first maps code into an intermediate semantic space.

More specifically:

- source files are decomposed into smaller code snippets
- each snippet is translated into a short functionality-oriented natural language description
- these functionality descriptions are embedded and clustered
- each repository is represented through the distribution of its functionality clusters
- classification is performed on top of that repository-level behavioral vector

This makes the pipeline more structured and more interpretable than a direct raw-code classification approach.

---

## High-level pipeline

The current pipeline is organized into four phases:

1. **Preprocessing and Chunking**
2. **Functionality Generation**
3. **Functionality Clustering**
4. **Repository Classification**

```text
Codebase
  -> Phase 1: Preprocessing and Chunking
      -> Repository files
      -> Code snippets
  -> Phase 2: Functionality Generation
      -> LLM
      -> NLP
      -> Code snippet functionalities
  -> Phase 3: Functionality Clustering
      -> Embeddings
      -> Clustering
      -> Functionality clusters
  -> Phase 4: Repository Classification
      -> Vectorization
      -> Repository vector
      -> Classification
```

For the full explanation and the updated diagram, see `docs/PIPELINE.md`.

---

## Main features

- multi-stage malware analysis pipeline based on semantic functionality extraction
- code chunking and snippet-based processing
- LLM-driven functionality generation for code snippets
- embedding-based functionality clustering
- repository-level vectorization based on clustered functionality patterns
- train/test execution modes for clustering and classification
- YAML-based modular configuration
- support for multiple programming languages through language-specific parsing configuration

---

## Project structure

The exact repository structure may evolve, but the project is conceptually organized around the following components:

- **CLI layer** for running the pipeline
- **configuration files** under `configs/`
- **dataset ingestion and preprocessing**
- **functionality generation**
- **clustering**
- **classification**
- **database-backed or pipeline-backed intermediate processing**, depending on the execution flow implemented in the codebase

---

## Quickstart

## Prerequisites

- Python 3.13+
- a virtual environment
- the dependencies required by the project
- valid environment variables in `.env`
- a configured dataset path in `configs/base.yaml`

Depending on your setup, you may also need:

- API credentials for the configured LLM provider
- the additional Python packages required by clustering and NLP

---

## Installation

Create and activate a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip` and install the project:

```bash
pip install -U pip
pip install -e .
```

Install any additional dependencies required by your pipeline stages if they are not already part of the environment.

---

## Environment configuration

The CLI loads environment variables from `.env` on startup.

Make sure your environment is configured correctly before running the pipeline. In particular, configure the credentials required by the LLM provider defined in `configs/llms.yaml`.

For example:

- if the provider is Google, configure the corresponding Google API key
- if the provider is OpenAI, configure the corresponding OpenAI API key

The exact variables depend on your implementation and environment setup.

---

## Dataset download

The dataset used in this project is not publicly distributed.

Access may be granted upon request for research purposes, subject to availability and approval.

For access requests, please contact:

- **Name**: Ioannis Moutevelidis  
- **Email**: <moutasjo@gmail.com>

## Dataset configuration

The dataset path is configured in `configs/base.yaml`.

Current example:

```yaml
paths:
  dataset: ".dataset/thesis_dataset"
```

The project expects a repository-based dataset organization. A typical dataset layout is:

```text
dataset/
  malware/
    repo_1/
    repo_2/
    ...
  benign/
    repo_a/
    repo_b/
    ...
```

If your local dataset uses different folder names or a different structure, adapt the pipeline and configuration accordingly.

---

## Running the CLI

Display the CLI help:

```bash
argus --help
```

Typical commands include:

```bash
argus load
argus split
argus generate_functionalities
argus cluster --mode train
argus classify --mode train
argus cluster --mode test
argus classify --mode test
```

If batch-based functionality generation is preferred:

```bash
argus generate_functionalities_batch_api
```

If cached functionality descriptions already exist:

```bash
argus load_cache_functionalities /path/to/cached_functionalities
```

See `docs/USAGE.md` for the full command reference and recommended workflows.

---

## Typical workflow

A standard end-to-end execution flow is:

```bash
# 1. Load the dataset
argus load

# 2. Generate snippet functionalities
argus generate_functionalities

# 3. Ensure functionality embeddings exist in Qdrant, then split
argus split

# 4. Train clustering
argus cluster --mode train

# 5. Train classification
argus classify --mode train

# 6. Run clustering on the test setup
argus cluster --mode test

# 7. Run classification on the test setup
argus classify --mode test
```

A shorter wrapper-based execution may also be available through:

```bash
argus train
argus test
```

Current wrapper behavior is:

```text
train -> cluster --mode train -> classify --mode train
test  -> cluster --mode test  -> classify --mode test
```

They do not run dataset loading, functionality generation, embedding creation, or splitting.

---

## Configuration files

The project now uses a broader and more modular configuration layout than the outdated version.

Current configuration files include:

- `configs/base.yaml`
- `configs/llms.yaml`
- `configs/split.yaml`
- `configs/clustering.yaml`
- `configs/classification.yaml`
- `configs/languages.yaml`
- `configs/important_nodes.yaml`
- `configs/comment_nodes.yaml`

### What they control

- `base.yaml` – application metadata, logging, and dataset path
- `llms.yaml` – LLM model configuration for snippet functionality generation
- `split.yaml` – dataset splitting parameters and binary labels
- `clustering.yaml` – clustering search/train/test settings
- `classification.yaml` – classifier train/test settings
- `languages.yaml` – supported programming languages and suffix mappings
- `important_nodes.yaml` – important AST node mappings used during structural processing
- `comment_nodes.yaml` – comment and annotation node mappings

See `docs/CONFIG.md` for full details.

---

## Current CLI commands

The central CLI currently exposes commands corresponding to the updated pipeline:

- `download_benign_dataset`
- `load`
- `split`
- `generate_functionalities`
- `generate_functionalities_batch_api`
- `load_cache_functionalities`
- `cluster --mode {search|train|test}`
- `classify --mode {train|test}`
- `train`
- `test`

This replaces the outdated README command set such as `setup`, `eval`, and `classify_1`, which no longer reflects the current CLI.

---

## Notes and caveats

- The README should be read together with `docs/USAGE.md`, `docs/CONFIG.md`, and `docs/PIPELINE.md`, since those files contain the authoritative detailed documentation.
- The current configuration shows a dedicated `snippet_processor` model in `llms.yaml`. If additional LLM processing stages are introduced later, their configuration should be documented there as well.
- Clustering and classification use explicit `train` and `test` sections in their respective YAML files.
- The current pipeline is centered on the four-phase thesis architecture rather than the older processor-centric description.
- If you reuse cached functionality descriptions, make sure they are compatible with the current clustering and classification setup.
- If you change the clustering dimensionality or the number of clusters, ensure that the classifier configuration remains consistent with that representation.
- `argus split` currently assumes embeddings already exist in Qdrant; see `docs/03_EMBEDDING_AND_SPLITTING.md`.

---

## Thesis positioning

Within the thesis, ARGUS is used as a framework for studying malware detection through **semantic behavioral abstraction**.

Its main research idea is that:

- code can first be translated into functionality-level natural language descriptions
- these descriptions can be organized into semantic clusters
- repositories can then be represented as distributions over these clusters
- malware detection can be performed on that higher-level behavioral representation

This separates low-level code syntax from higher-level behavior modeling and supports a more interpretable repository classification process.

---

## Where to start

For a new user, the recommended reading order is:

1. `docs/SETUP.md`
2. `docs/CONFIG.md`
3. `docs/USAGE.md`
4. `docs/PIPELINE.md`
5. `docs/README.md` for the complete documentation index

That sequence gives the cleanest path from installation to execution to understanding the system design.
