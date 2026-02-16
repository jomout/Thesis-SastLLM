# Configuration

SASTLLM uses three YAML files located under `configs/`.

## base.yaml

```yaml
app:
  name: sastllm

log:
  level: INFO
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  file: logs/sastllm.log

paths:
  database_dir: "./.dataset"     # local dataset root (contains malware/ and benignware/)
  evaluation_dir: "./.csn_dataset"  # CSN storage (optional)
```

Notes:

- `database_dir` drives `sastllm setup` — repository ingestion and snippet creation.
- `evaluation_dir` is used by `sastllm setup_eval` (CodeSearchNet download).

## llms.yaml

```yaml
app:
  name: sastllm

models:
  snippet_processor:
    host: "google"              # or "openai"
    name: "gemini-2.5-flash"    # model name in the provider
    params:
      temperature: 0
      max_tokens: null
      timeout: null
      max_retries: 5

  cluster_processor:
    host: "google"
    name: "gemini-2.5-flash"
    params:
      temperature: 0

  file_processor:
    host: "google"
    name: "gemini-2.5-flash"
    params:
      temperature: 0

```

Notes:

- The CLI constructs LangChain chat models from these entries for intermediate analysis (e.g., snippet and cluster processing).
- Ensure your environment has `GOOGLE_API_KEY` (for `host: google`) or `OPENAI_API_KEY` (for `host: openai`).
- Repository classification is ML-only.

## classification.yaml

```yaml
app:
  name: sastllm

classification:
  train:
    name: "mlp-demo"
    directory: "./models"
    params:
      lr: 0.001
      weight_decay: 0.001
      l1_param: 0.001
      validation_ratio: 0.2
      test_ratio: 0.2
      epochs: 100
      batch_size: 50

  inference:
    name: "mlp-demo"
    directory: "./models"
    params:
      lr: 0.001
      weight_decay: 0.001
      l1_param: 0.001
      validation_ratio: 0
      test_ratio: 1
      epochs: 100
      batch_size: 50
```

Notes:

- The `RepositoryClassifier` (ML route) reads `classification.<mode>.params` and threads them into Lightning components.
- The training path is partially commented in code — see docs/PIPELINE.md for what remains to enable end-to-end ML.
