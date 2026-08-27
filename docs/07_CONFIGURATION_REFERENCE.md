# Configuration Reference

The project uses YAML files under `configs/` plus environment variables loaded from `.env`.

## YAML loading

The helper `scripts.utils.load_yaml()` loads YAML files as dictionaries and raises `FileNotFoundError` if the path does not exist.

## `base.yaml`

Current content:

```yaml
app:
  name: argus

log:
  level: INFO
  file: logs/argus.log
  file_level: DEBUG
  max_bytes: 10485760
  backup_count: 5

paths:
  dataset: ".dataset/thesis_dataset"
```

Used by:

- logging setup
- `argus load`

`level` controls the structured console threshold. `file_level` controls the rotating JSON-lines file independently. `max_bytes` and `backup_count` bound retained log volume. Use `INFO` on the console and `DEBUG` in the file for normal experiments; temporarily set the console to `DEBUG` when investigating batch, parser, encoder-shape, or storage behavior.

## `llms.yaml`

Current content:

```yaml
app:
  name: argus

models:
  snippet_processor:
    host: "google"
    name: "gemini-2.5-flash"
    params:
      temperature: 0
      max_tokens: null
      timeout: null
      max_retries: 5
```

Used by:

- `argus generate_functionalities`

Only `snippet_processor` is accepted by the current `get_model()` factory.

## `split.yaml`

Current content:

```yaml
split:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  training:
    ratio: 0.7

  testing:
    ratio: 0.3

  binary_labels:
    - 0: "benign"
    - 1: "malicious"
```

Used by:

- dataset splitting
- Qdrant collection naming
- binary label mapping in repository encoding

The train and test ratios must sum to `1.0`.

## `clustering.yaml`

Current content:

```yaml
clustering:
  evaluation:
    sample_size: 100000
    silhouette_sample_size: 5000
    silhouette_samples_per_cluster: 5
    silhouette_metric: "euclidean"
    random_state: 42
    elbow_window_factor: 2.0
    max_silhouette_singleton_fraction: 0.5

  search:
    grid_search: [1908726]
    save_model_dir: "models/clustering/searching_models"
    random_state: 42
    batch_size: 1000
    min_samples_per_cluster: 20
    num_k_candidates: 30

  train:
    k: 10661
    save_model_dir: "models/clustering/trained_models"

  test:
    k: 10661
    load_model_dir: "models/clustering/trained_models/clusterer_10661_<timestamp>"
```

Used by:

- `argus cluster --mode search`
- `argus cluster --mode train`
- `argus cluster --mode test`

Note: `search.random_state` is passed into the current `MiniBatchKMeansClusterer` implementation. `test.load_model_dir` must contain `model.joblib`, `model.onnx`, and their checksummed `manifest.json`; test mode restores `model.joblib` after bundle verification.

The `evaluation` section controls the fixed candidate-comparison reservoir, cluster-stratified silhouette, elbow neighborhood, and silhouette reliability threshold. `sample_size` should exceed the expected K so the reservoir can also initialize MiniBatchKMeans independently of source order. `silhouette_samples_per_cluster` sets the target minimum representation for each selected silhouette cluster. Search and train create timestamped run directories under their respective `save_model_dir`; every report and plot is stored beside its model.

## `classification.yaml`

Current content:

```yaml
models:
  mlp:
    hidden_dims: [512, 256]
    dropout: 0.2
  lstm:
    embedding_dim: 128
    hidden_dim: 128
    num_layers: 3
    dropout: 0.2
    bidirectional: false
    pooling: "last"
    max_sequence_length: 512
    truncation: "first"
  transformer:
    input_encoding: "cluster_distribution"
    embedding_dim: 128
    num_layers: 2
    num_heads: 4
    feedforward_dim: 256
    dropout: 0.2
    pooling: "mean"
    max_sequence_length: 256
    truncation: "first"

classification:
  train:
    model: lstm
    save_model_dir: "models/classification/trained_models"
    params:
      k: 10661
      lr: 0.0005
      weight_decay: 0.0005
      l1_param: 0.005
      epochs: 30
      batch_size: 64
      seed: 42

  test:
    model: lstm
    load_model_dir: "models/classification/trained_models/model_20260426_204737"
    params:
      k: 10661
      lr: 0.0005
      weight_decay: 0.0005
      l1_param: 0.005
      batch_size: 64
      seed: 42
```

Used by:

- `argus classify --mode search`
- `argus classify --mode train`
- `argus classify --mode test`

Important naming note: `l1_param` is accepted as an alias for the runtime field `l1_lambda`.

Classification does not accept `clusterer_manifest`. It consumes cluster ids already stored in PostgreSQL and uses `params.k` to define the expected cluster vocabulary size.

Model architecture fields live in strict reusable profiles under `models`. Each classification mode selects one profile by name. Optimization and data-loading fields remain under the mode's `params` mapping.

For `models.transformer.input_encoding`, use `"ordered_tokens"` to preserve functionality order or `"cluster_distribution"` to attend over aggregate cluster frequencies. In distribution mode, `max_sequence_length` limits the most frequent nonzero clusters included per repository and `pooling` must be `"mean"` or `"max"`.

## `languages.yaml`

Defines Tree-sitter grammar repositories and supported suffixes.

Used by:

- file discovery
- parser creation
- prompt AST extraction
- comment stripping

Current language set:

```text
asm, c, cpp, c_sharp, java, javascript, typescript, tsx,
python, ruby, go, php, perl, cuda, html, pascal, powershell,
bash, rust, swift, css, vb_dotnet
```

## `important_nodes.yaml`

Maps language-specific Tree-sitter node types to semantic labels.

Used by:

- AST-aware chunk breakpoints

Examples:

- Python: `class_definition`, `function_definition`
- C/C++: `function_definition`, `struct_specifier`, preprocessor nodes
- Java: package/import/class/interface/enum/method/constructor declarations
- JavaScript/TypeScript: imports, exports, classes, functions, arrow functions

## `comment_nodes.yaml`

Maps language-specific comment and decorator node types.

Used by:

- moving chunk boundaries to include nearby comment blocks
- stripping comments during chunk generation

## Environment variables

Loaded through `.env` and `pydantic-settings`.

Common variables:

```text
GOOGLE_API_KEY
OPENAI_API_KEY
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_DB
POSTGRES_HOST
POSTGRES_PORT
ENDPOINT_URL
ACCESS_TOKEN
```

`ENDPOINT_URL` and `ACCESS_TOKEN` are used only by the custom `issel` LLM host.
