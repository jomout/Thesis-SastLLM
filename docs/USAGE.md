# Usage

This document explains common CLI flows provided by `scripts/cli.py`.

## CLI overview

```zsh
sastllm --help
```

Commands:

- `setup`        → Load dataset into DB (repositories/files/snippets)
- `train`        → Generate snippet functionalities (LLM) and cluster tags
- `classify`     → Classify repositories via ML
- `eval`         → Evaluate using CodeSearchNet (CSN)
- `clear`        → Clear the database tables
- `setup_eval`   → Download CSN split and insert into DB

## Typical workflows

### 1) End-to-end pipeline (ML classification)

```zsh
# 1) Insert dataset into DB
sastllm setup

# 2) Generate snippet functionalities via LLM
sastllm train

# 3) Classify repositories via ML
sastllm classify

# Output: ML model logs under models/ and classification results reported by the CLI
```

### 2) End-to-end with ML classification

```zsh
sastllm setup
sastllm train
sastllm classify_1
```

Notes:

- The ML classifier path is the supported route. Ensure clustering completed before running classification.

### 3) Evaluation with CodeSearchNet

```zsh
# Download CSN split and insert into DB tables
sastllm setup_eval

# Compare your snippet functionalities with CSN docstrings (prints similarities)
sastllm eval
```

## Data layout expectations

Your dataset root (from `configs/base.yaml: paths.database_dir`) should look like:

```text
/path/to/dataset/
  malware/
    repo1/
      <source files>
  benignware/
    repoA/
      <source files>
```

## Logging

Logs are configured via `configs/base.yaml` and written to `logs/sastllm.log`.
Set the log level and format there. Console logs are always enabled.
