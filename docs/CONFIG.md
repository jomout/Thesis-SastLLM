# Configuration

This document describes the configuration files currently used by **SAST-LLM** and their role in the thesis pipeline.

The framework uses multiple YAML configuration files under the `configs/` directory. Each file controls a different stage of the pipeline, including dataset paths, language support, AST node handling, LLM settings, dataset splitting, clustering, and classification.

## Configuration files overview

The current configuration is organized into the following files:

- `base.yaml`
- `llms.yaml`
- `split.yaml`
- `clustering.yaml`
- `classification.yaml`
- `languages.yaml`
- `important_nodes.yaml`
- `comment_nodes.yaml`

---

## `base.yaml`

This file contains the general application and path-level configuration.

```yaml
app:
  name: sastllm

log:
  level: INFO
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  file: logs/sastllm.log

paths:
  dataset: "../test_pipeline/test_dataset"
```

### Purpose

`base.yaml` defines:

- the application name
- the logging configuration
- the base dataset path used by the pipeline

### Notes

- The application is named `sastllm`.
- Logging is configured with level `INFO`, a timestamped format, and output file `logs/sastllm.log`.
- The current dataset path is stored under `paths.dataset` and points to `../test_pipeline/test_dataset`.
- This replaces the older `database_dir` and `evaluation_dir` structure used in the outdated documentation. The current configuration now exposes a single dataset path instead.

---

## `llms.yaml`

This file defines the LLM configuration used in the functionality generation stage.

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

### Purpose

`llms.yaml` defines the model used for snippet-level functionality generation.

### Notes

- The configured model is `snippet_processor`.
- Its provider is `google`, and the configured model name is `gemini-2.5-flash`.
- The current parameters are:
  - `temperature: 0`
  - `max_tokens: null`
  - `timeout: null`
  - `max_retries: 5`
- Unlike the outdated configuration, the currently provided file does **not** define separate `cluster_processor` or `file_processor` models. Only `snippet_processor` is present in the uploaded configuration.

---

## `split.yaml`

This file controls the dataset split configuration and label mapping.

```yaml
split:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  training:
    ratio: 0.0

  testing:
    ratio: 1.0

  binary_labels:
    - 0: "benign"
    - 1: "malicious"
```

### Purpose

`split.yaml` defines how the dataset split stage is parameterized.

### Notes

- The configured embedding model for splitting is `sentence-transformers/all-mpnet-base-v2`.
- The current uploaded configuration sets:
  - training ratio to `0.0`
  - testing ratio to `1.0`
- The binary label mapping is:
  - `0 -> benign`
  - `1 -> malicious`
- Since the current file assigns the full ratio to testing, this appears to reflect either a testing-only setup or an experimental placeholder configuration. That interpretation is an inference from the file contents.

---

## `clustering.yaml`

This file defines the configuration for clustering search, training, and test execution.

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

### Purpose

`clustering.yaml` controls the clustering stage for the three supported modes:

- `search`
- `train`
- `test`

### Notes

- In `search` mode:
  - `grid_search` is set to `[1908726]`
  - models are saved under `models/clustering/searching_models`
  - plots are saved under `plots/clustering/searching`
  - `random_state` is `42`
- In `train` mode:
  - `k` is set to `10661`
  - trained models are saved under `models/clustering/trained_models`
- In `test` mode:
  - `k` is also `10661`
  - the clustering model is loaded from `models/clustering/trained_models/clusterer_k_10661.joblib`
- This aligns with the CLI modes `cluster --mode search`, `cluster --mode train`, and `cluster --mode test`. That mapping is consistent with the current CLI design and the configuration structure.

---

## `classification.yaml`

This file defines the classifier configuration for train and test modes.

```yaml
classification:

  train:
    save_model_dir: "models/classification/trained_models"
    params:
      k: 10661
      lr: 0.0005
      weight_decay: 0.001
      l1_param: 0.001
      epochs: 30
      batch_size: 16
      seed: 42

  test:
    load_model_dir: "models/classification/trained_models/binary_classifier_model"
    params:
      k: 10661
      lr: 0.0005
      weight_decay: 0.001
      l1_param: 0.001
      batch_size: 16
      seed: 42
```

### Purpose

`classification.yaml` controls the repository classification stage in both training and testing modes.

### Notes

- In `train` mode:
  - trained models are saved to `models/classification/trained_models`
  - the parameters are:
    - `k: 10661`
    - `lr: 0.0005`
    - `weight_decay: 0.001`
    - `l1_param: 0.001`
    - `epochs: 30`
    - `batch_size: 16`
    - `seed: 42`
- In `test` mode:
  - the model is loaded from `models/classification/trained_models/binary_classifier_model`
  - the parameters are:
    - `k: 10661`
    - `lr: 0.0005`
    - `weight_decay: 0.001`
    - `l1_param: 0.001`
    - `batch_size: 16`
    - `seed: 42`
- Compared with the outdated documentation, the current file no longer uses `name`, `directory`, `validation_ratio`, or `test_ratio` fields. Instead, it uses explicit save/load directories and a smaller parameter set.

---

## `languages.yaml`

This file defines the supported programming languages, their Tree-sitter grammar sources, and the filename suffixes used for matching.

### Purpose

`languages.yaml` determines which programming languages the ingestion and parsing stages can recognize.

### Notes

The current configuration includes support for the following languages: `asm`, `c`, `cpp`, `c_sharp`, `java`, `javascript`, `typescript`, `tsx`, `python`, `ruby`, `go`, `php`, `perl`, `cuda`, `html`, `pascal`, `powershell`, `bash`, `rust`, `swift`, `css`, and `vb_dotnet`.

Each entry specifies:

- the internal language name
- the Tree-sitter repository
- optional `subdir` information
- the supported file suffixes

Examples:

- `python` uses the Tree-sitter Python grammar and matches `.py` files.
- `javascript` matches `.js` and `.jsx`.
- `typescript` matches `.ts`, while `tsx` matches `.tsx`.
- `c` matches `.c` and `.h`, while `cpp` matches `.cpp` and `.hpp`.

This file is part of the multi-language support of the thesis pipeline.

---

## `important_nodes.yaml`

This file defines the AST node types considered important during structural analysis and chunking, grouped by programming language.

### Purpose

`important_nodes.yaml` provides language-specific mappings between Tree-sitter node names and higher-level semantic labels such as `Function`, `Class`, `Method`, `Struct`, `Import`, and related constructs.

### Notes

Examples from the current configuration include:

- For **C**:
  - `function_definition -> Function`
  - `struct_specifier -> Struct`
  - `enum_specifier -> Enum`
  - preprocessor nodes such as `preproc_include`, `preproc_ifdef`, and `preproc_def` are mapped to labels such as `Include`, `Ifdef`, and `Define`.
- For **Python**:
  - `import_statement -> Import`
  - `class_definition -> Class`
  - `function_definition -> Function`
- For **JavaScript** and **TypeScript**:
  - import/export, classes, functions, arrow functions, and statement blocks are explicitly mapped.
- For **Java**, **C#**, **Go**, **PHP**, **Rust**, **Swift**, and others, the file specifies analogous important node mappings.

This file is central to the structural representation used during preprocessing and chunk extraction.

---

## `comment_nodes.yaml`

This file defines the AST node types used to identify comments and related metadata constructs per language.

### Purpose

`comment_nodes.yaml` allows the system to identify comment nodes and selected metadata-style nodes, depending on the programming language.

### Notes

- Most languages map a `comment`-type key to the label `Comment`.
- Some languages include additional constructs:
  - `javascript`, `typescript`, `tsx`, and `python` include `decorator: Decorator` in addition to comments.
  - `php` includes `attribute: Attribute`.
- Supported languages include `asm`, `c`, `cpp`, `c_sharp`, `java`, `javascript`, `typescript`, `tsx`, `python`, `ruby`, `go`, `php`, `perl`, `cuda`, `html`, `pascal`, `powershell`, `bash`, `rust`, `swift`, `css`, and `vb_dotnet`.

This configuration is useful when separating code content from comments or language-specific annotations during preprocessing.

---

## How the configuration maps to the pipeline

The configuration files correspond to the thesis pipeline as follows:

| Pipeline phase | Main configuration files |
| --- | --- |
| Preprocessing and chunking | `base.yaml`, `languages.yaml`, `important_nodes.yaml`, `comment_nodes.yaml` |
| Functionality generation | `llms.yaml` |
| Dataset splitting | `split.yaml` |
| Functionality clustering | `clustering.yaml` |
| Repository classification | `classification.yaml` |

This reflects the current modular structure of the system.

---

## Practical remarks

- The old `CONFIG.md` is outdated because it documents only three files: `base.yaml`, `llms.yaml`, and `classification.yaml`.
- The current configuration is broader and includes explicit files for splitting, clustering, language registration, important AST nodes, and comment nodes.
- The current configuration also uses `train` and `test` sections consistently in both clustering and classification.  
- The value `k: 10661` appears in both clustering and classification, indicating that the classifier expects a feature space compatible with the clustering setup. This is a grounded inference based on the shared configuration value.  

---

## Summary

The current SAST-LLM configuration is no longer limited to a small set of generic YAML files. It now uses a more explicit and modular structure:

- `base.yaml` for application paths and logging
- `llms.yaml` for snippet-generation model settings
- `split.yaml` for data partitioning and binary labels
- `clustering.yaml` for clustering search/train/test modes
- `classification.yaml` for classifier train/test modes
- `languages.yaml` for supported programming languages
- `important_nodes.yaml` for AST node importance mappings
- `comment_nodes.yaml` for comment and annotation node mappings

This updated structure better reflects the current thesis pipeline and the multi-stage design of the framework.
