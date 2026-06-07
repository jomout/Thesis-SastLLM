# Configuration Reference

The project uses YAML files under `configs/` plus environment variables loaded from `.env`.

## YAML loading

The helper `scripts.utils.load_yaml()` loads YAML files as dictionaries and raises `FileNotFoundError` if the path does not exist.

## `base.yaml`

Current content:

```yaml
app:
  name: sastllm

log:
  level: INFO
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  file: logs/sastllm.log

paths:
  dataset: ".dataset/thesis_dataset"
```

Used by:

- logging setup
- `sastllm load`

## `llms.yaml`

Current content:

```yaml
app:
  name: sastllm

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

- `sastllm generate_functionalities`

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
  search:
    grid_search: [1908726]
    save_model_dir: "models/clustering/searching_models"
    save_plots_dir: "plots/clustering/searching"
    random_state: 42

  train:
    k: 10661
    save_model_dir: "models/clustering/trained_models"

  test:
    k: 10661
    load_model_file: "models/clustering/trained_models/clusterer_k_10661.joblib"
```

Used by:

- `sastllm cluster --mode search`
- `sastllm cluster --mode train`
- `sastllm cluster --mode test`

Note: `search.random_state` is passed into the current `MiniBatchKMeansClusterer` implementation.

## `classification.yaml`

Current content:

```yaml
classification:
  train:
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

- `sastllm classify --mode search`
- `sastllm classify --mode train`
- `sastllm classify --mode test`

Important naming note: `l1_param` is accepted as an alias for the runtime field `l1_lambda`.

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
