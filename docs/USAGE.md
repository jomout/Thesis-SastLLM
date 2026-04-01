# Usage

This document describes the command-line interface of the **SAST-LLM** framework and the recommended execution flows used in the thesis experiments.

## CLI overview

Display the available commands and options:

```bash
sastllm --help
```

The CLI initializes the project environment on startup by:

- loading environment variables from `.env`
- configuring application logging
- preparing the runtime for pipeline execution

## Available commands

### Dataset preparation

#### `download_benign_dataset`

Downloads the benign dataset used in the experiments.

```bash
sastllm download_benign_dataset
```

This command downloads the **CodeSearchNet** dataset and stores it in the configured local dataset directory.

---

#### `load`

Loads the local dataset into the database.

```bash
sastllm load
```

This command parses the local repository dataset and inserts the corresponding records into the database, including repository-level and snippet-level information.

---

#### `split`

Splits the dataset into the required subsets.

```bash
sastllm split
```

This command performs dataset splitting for the experimental pipeline, typically producing the training, validation, and test partitions required by later stages.

---

### Functionality generation

#### `generate_functionalities`

Generates functionality descriptions for code snippets using the configured LLM.

```bash
sastllm generate_functionalities
```

This command processes code snippets stored in the database and produces short functionality descriptions that are later used for semantic clustering.

---

#### `generate_functionalities_batch_api`

Generates functionality descriptions using the OpenAI Batch API.

```bash
sastllm generate_functionalities_batch_api
```

This command is intended for larger-scale functionality generation. It prepares batch requests, submits them to the API, and processes the returned outputs back into the local pipeline.

---

#### `load_cache_functionalities`

Loads previously generated functionality descriptions from a local directory.

```bash
sastllm load_cache_functionalities /path/to/cached_functionalities
```

Use this command when functionality descriptions have already been generated and stored externally, and should be imported without repeating the LLM inference step.

---

### Clustering

#### `cluster`

Clusters the generated functionality descriptions.

```bash
sastllm cluster --mode train
```

Available modes:

- `train`
- `test`
- `search`

Examples:

```bash
sastllm cluster --mode train
sastllm cluster --mode test
sastllm cluster --mode search
```

The selected mode determines which subset or operational scenario is used during clustering.

---

### Classification

#### `classify`

Runs the repository classification stage.

```bash
sastllm classify --mode train
```

Available modes:

- `train`
- `test`

Examples:

```bash
sastllm classify --mode train
sastllm classify --mode test
```

This command performs repository-level classification based on the functionality-cluster representation produced by the previous stages.

---

### End-to-end pipelines

#### `train`

Runs the full training pipeline.

```bash
sastllm train
```

This command executes the training-stage pipeline as defined in the project implementation.

---

#### `test`

Runs the full testing pipeline.

```bash
sastllm test
```

This command executes the testing-stage pipeline as defined in the project implementation.

---

## Recommended execution flows

## 1. Full experimental pipeline from raw dataset

This is the standard workflow when starting from the repository dataset.

```bash
# 1. Download benign repositories if needed
sastllm download_benign_dataset

# 2. Load the dataset into the database
sastllm load

# 3. Split the dataset
sastllm split

# 4. Generate snippet-level functionality descriptions
sastllm generate_functionalities
# or, for large-scale generation:
# sastllm generate_functionalities_batch_api

# 5. Cluster functionality descriptions
sastllm cluster --mode train

# 6. Train the classification stage
sastllm classify --mode train

# 7. Evaluate on the test split
sastllm cluster --mode test
sastllm classify --mode test
```

This workflow corresponds to the main thesis pipeline:

1. dataset ingestion  
2. dataset splitting  
3. functionality generation  
4. semantic clustering  
5. repository-level classification  

---

## 2. Training and testing through the pipeline wrappers

If the project configuration already encapsulates the individual stages, the higher-level commands may be used instead.

```bash
sastllm train
sastllm test
```

This is the most compact way to reproduce the predefined training and testing workflows implemented in the codebase.

---

## 3. Reusing cached functionality descriptions

If functionality descriptions have already been generated in a previous run, they can be reloaded directly.

```bash
sastllm load_cache_functionalities /path/to/cached_functionalities
sastllm cluster --mode train
sastllm classify --mode train
```

This avoids repeating the LLM generation stage and is useful for reproducibility experiments or repeated clustering and classification trials.

---

## Example command sequence used in practice

A typical sequence for a complete experiment is the following:

```bash
sastllm load
sastllm split
sastllm generate_functionalities
sastllm cluster --mode train
sastllm classify --mode train
sastllm cluster --mode test
sastllm classify --mode test
```

If batch-based functionality generation is preferred:

```bash
sastllm load
sastllm split
sastllm generate_functionalities_batch_api
sastllm cluster --mode train
sastllm classify --mode train
sastllm cluster --mode test
sastllm classify --mode test
```

## Input and data assumptions

The CLI assumes that:

- the local dataset paths are correctly configured in the project configuration files
- the required database is accessible
- the `.env` file contains the necessary environment variables
- snippet extraction and repository loading are supported by the dataset format used in the thesis experiments

A typical dataset structure is expected to follow a repository-based organization such as:

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

The exact paths are determined by the project configuration.

## Logging

Logging is initialized automatically when the CLI starts.

Logs provide visibility into:

- project initialization
- dataset loading
- functionality generation
- clustering
- classification
- training and testing pipelines

The logger used by the CLI is obtained through the project logging utilities:

- `setup_logging()`
- `get_logger(__name__)`

## Notes for thesis usage

For the purposes of the thesis, the commands can be grouped into the following conceptual stages:

| Thesis Stage | CLI Commands |
| --- | --- |
| Dataset acquisition | `download_benign_dataset` |
| Dataset ingestion | `load` |
| Dataset partitioning | `split` |
| Functionality extraction | `generate_functionalities`, `generate_functionalities_batch_api`, `load_cache_functionalities` |
| Semantic clustering | `cluster --mode ...` |
| Repository classification | `classify --mode ...` |
| Wrapped execution | `train`, `test` |

## Minimal command reference

```bash
sastllm download_benign_dataset
sastllm load
sastllm split
sastllm generate_functionalities
sastllm generate_functionalities_batch_api
sastllm load_cache_functionalities /path/to/dir
sastllm cluster --mode train
sastllm cluster --mode test
sastllm cluster --mode search
sastllm classify --mode train
sastllm classify --mode test
sastllm train
sastllm test
```
