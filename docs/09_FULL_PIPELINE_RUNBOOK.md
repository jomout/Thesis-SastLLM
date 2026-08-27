# Full Pipeline Runbook

This is the command-first guide for running a complete SAST-LLM experiment, from raw repositories to final classification metrics.

For detailed stage behavior, use [PIPELINE.md](./PIPELINE.md). For clustering research decisions, use [08_CLUSTERING_RESEARCH_GUIDE.md](./08_CLUSTERING_RESEARCH_GUIDE.md).

## Pipeline at a glance

```text
1. Configure experiment
2. Start PostgreSQL and Qdrant
3. Load repositories and create snippets
4. Generate functionality tags
5. Embed functionality tags into Qdrant
6. Assign train/test splits
7. Search for and validate clustering K
8. Fit the final clusterer and assign all cluster ids
9. Inspect repository encodings
10. Search classifier hyperparameters
11. Train the final classifier
12. Evaluate the frozen classifier on test data
```

Run commands from the project root.

## 1. Configure the experiment

Review these files before starting:

| File                          | Configure                                              |
| ----------------------------- | ------------------------------------------------------ |
| `configs/base.yaml`           | source dataset path and logging                        |
| `configs/llms.yaml`           | functionality-generation model                         |
| `configs/languages.yaml`      | supported source languages                             |
| `configs/split.yaml`          | embedding model and train/test ratios                  |
| `configs/clustering.yaml`     | K search, clustering evaluation, and model paths       |
| `configs/classification.yaml` | architecture, K, training params, and checkpoint paths |

Create or verify `.env` with the PostgreSQL and model-provider credentials described in [SETUP.md](./SETUP.md).

Install the project:

```bash
uv sync
uv run argus --help
```

## 2. Start storage services

```bash
docker compose up -d
docker compose ps
```

Expected services:

- PostgreSQL stores repositories, files, snippets, functionalities, splits, and cluster ids.
- Qdrant stores functionality embeddings and split payloads.

Do not continue until both services are healthy and reachable using `.env`.

## 3. Load the dataset

Set the local repository dataset in `configs/base.yaml`:

```yaml
paths:
  dataset: ".dataset/thesis_dataset"
```

Then run:

```bash
uv run argus load
```

Expected result:

- repositories are inserted into PostgreSQL
- source files are discovered
- source code is chunked into snippets

This stage is normally run once for a dataset. Check the logs before continuing; an empty or partially loaded dataset invalidates every later count.

## 4. Generate functionalities

Choose one functionality source.

### Direct LLM generation

```bash
uv run argus generate_functionalities
```

### Existing cached results

```bash
uv run argus load_cache_functionalities /path/to/cached_functionalities
```

Expected result:

- functionality rows exist for processed snippets
- each functionality has a description and normalized tag
- processed status propagates through snippets, files, and repositories

The Batch API command downloads raw result files but does not currently parse them into PostgreSQL automatically. Do not proceed from Batch API generation until its results have been imported.

## 5. Create functionality embeddings

Select the embedding model in `configs/split.yaml`:

```yaml
split:
  model_name: "sentence-transformers/all-mpnet-base-v2"
```

There is currently no dedicated embedding CLI command. Run the existing embedding helper:

```bash
uv run python -c 'from scripts.utils import load_yaml; from argus.utils.dataset_splitter import DatasetSplitter; c = load_yaml("configs/split.yaml")["split"]; DatasetSplitter(model_name=c["model_name"]).embed_all_repositories()'
```

Expected result:

- one Qdrant point exists per functionality
- point id equals `functionality_id`
- payload contains `repository_id`, `tag`, and `split: full`
- the Qdrant collection name is the model name with `/` replaced by `_`

The embedder skips ids already present in the collection. If the functionality tag or embedding model changes, use a new collection or deliberately rebuild the existing one.

## 6. Assign train/test splits

Confirm the ratios in `configs/split.yaml`, then run:

```bash
uv run argus split
```

Expected result:

- `repositories.split` is populated in PostgreSQL
- Qdrant payloads change from `split: full` to `split: train` or `split: test`
- splitting is stratified by repository label with random seed `42`

Important: this command assumes embeddings already exist. It does not create missing vectors.

## 7. Search for clustering K

Review the evaluation and search sections of `configs/clustering.yaml`:

```yaml
clustering:
  evaluation:
    sample_size: 100000
    silhouette_sample_size: 5000
    silhouette_samples_per_cluster: 5
    random_state: 42

  search:
    grid_search: [1908726]
    min_samples_per_cluster: 20
    num_k_candidates: 30
```

Run:

```bash
uv run argus cluster --mode search
```

Inspect:

```text
models/clustering/searching_models/clusterers_<n>_<timestamp>/
  clusterer_<n>_<k>_<timestamp>/
    model.joblib
    model.onnx
    manifest.json
    clusterer_<n>_<k>_<timestamp>_quality.json
    clusterer_<n>_<k>_<timestamp>_selection_quality.json
    clusterer_<n>_<k>_<timestamp>_selection_candidates.csv
    clusterer_<n>_<k>_<timestamp>_selection_quality.png
```

Do not accept K from inertia alone. Review:

- low cohesion RMS
- high separation RMS
- reliable, comparatively high silhouette
- strong Calinski-Harabasz score
- a clear inertia elbow
- few empty, singleton, and undersized clusters

The complete interpretation and reporting procedure is in [08_CLUSTERING_RESEARCH_GUIDE.md](./08_CLUSTERING_RESEARCH_GUIDE.md).

## 8. Synchronize the selected K

Suppose search selects `K_SELECTED`.

Update `configs/clustering.yaml`:

```yaml
clustering:
  train:
    k: K_SELECTED

  test:
    k: K_SELECTED
    load_model_dir: "models/clustering/trained_models/clusterer_K_SELECTED_TIMESTAMP"
```

Update all modes in `configs/classification.yaml`:

```yaml
classification:
  search:
    params:
      k: K_SELECTED
  train:
    params:
      k: K_SELECTED
  test:
    params:
      k: K_SELECTED
```

Replace `K_SELECTED` with the integer value. All stages must use the same cluster vocabulary size.

## 9. Fit and apply the final clusterer

Train on the train split:

```bash
uv run argus cluster --mode train
```

Inspect the resulting quality report before continuing:

```text
models/clustering/trained_models/clusterer_<k>_<timestamp>/clusterer_<k>_<timestamp>_quality.json
```

Confirm that the evaluated count and cluster-size statistics are plausible. Train mode also writes cluster ids for train functionalities.

Apply the frozen model to test functionalities:

```bash
uv run argus cluster --mode test
```

After this step, both train and test functionality rows should have zero-based cluster ids.

## 10. Inspect repository encodings

Validate at least one train and one test repository before classifier training.

For the MLP distribution encoding:

```bash
uv run argus-inspect-distribution --repository-id 123
```

For the ordered LSTM/Transformer token encoding:

```bash
uv run argus-inspect-timeseries --repository-id 123
```

Verify that:

- the expected files and functionalities appear
- functionality ids define sequence order
- cluster ids are in `[0, K_SELECTED - 1]`
- distribution counts match the printed functionalities
- padding and truncation match the selected sequence model configuration

## 11. Select the classifier architecture

Set `classification.<mode>.model` consistently in `configs/classification.yaml`:

| Model         | Encoder                        | Use case                       |
| ------------- | ------------------------------ | ------------------------------ |
| `mlp`         | cluster distribution           | unordered frequency baseline   |
| `lstm`        | ordered cluster-token sequence | sequential baseline            |
| `transformer` | ordered cluster-token sequence | attention-based sequence model |

For class imbalance, begin with only one correction:

```yaml
use_weighted_sampler: true
use_class_weights: false
```

Using both can overcorrect the minority class.

## 12. Search classifier hyperparameters

Configure `classification.search.grid_search`, then run:

```bash
uv run argus classify --mode search
```

Expected outputs:

```text
models/classification/searching_models/<search_run>/
plots/classification/searching/<search_run>_accuracy.png
plots/classification/searching/search_summary.json
```

Each search model directory includes train/validation predictions and metrics. Select parameters using validation results, not test metrics.

Copy the selected runtime values into `classification.train.params` and keep the same model profile.

## 13. Train the final classifier

```bash
uv run argus classify --mode train
```

Expected output directory:

```text
models/classification/trained_models/model_<timestamp>/
```

It contains the Lightning checkpoint, PyTorch weights, ONNX model, checksummed manifest, effective configuration, metadata, train predictions, and train metrics.

The three model files are `model.ckpt`, `model.pt`, and `model.onnx`. Test mode restores `model.ckpt` from the configured directory.

Record the printed model directory. Set it in `classification.test.load_model_dir` and keep `classification.test.model` consistent with training:

```yaml
classification:
  test:
    model: transformer
    load_model_dir: "models/classification/trained_models/model_<timestamp>"
```

## 14. Evaluate on the test split

Run the frozen classifier exactly once for the final reported experiment:

```bash
uv run argus classify --mode test
```

Expected files in the trained model directory:

```text
test_predictions.json
test_metrics.json
```

Report more than accuracy for an imbalanced dataset. Include at least the confusion matrix and class-specific precision, recall, and F1 values available in the generated metrics.

## Complete command sequence

For a fresh dataset:

```bash
docker compose up -d
uv sync

uv run argus load
uv run argus generate_functionalities

uv run python -c 'from scripts.utils import load_yaml; from argus.utils.dataset_splitter import DatasetSplitter; c = load_yaml("configs/split.yaml")["split"]; DatasetSplitter(model_name=c["model_name"]).embed_all_repositories()'
uv run argus split

uv run argus cluster --mode search
# Review clustering reports and synchronize K in clustering.yaml and classification.yaml.

uv run argus cluster --mode train
uv run argus cluster --mode test

uv run argus-inspect-distribution --repository-id 123
uv run argus-inspect-timeseries --repository-id 123

uv run argus classify --mode search
# Copy the selected search parameters into classification.train.params.

uv run argus classify --mode train
# Point classification.test.load_model_dir to the newly trained model directory.

uv run argus classify --mode test
```

## Prepared-data sequence

When PostgreSQL already contains processed functionalities, Qdrant already contains split embeddings, and K is already validated:

```bash
uv run argus cluster --mode train
uv run argus cluster --mode test
uv run argus classify --mode train
uv run argus classify --mode test
```

After all paths and parameters are finalized, the wrappers provide the same final-stage sequence:

```bash
uv run argus train
uv run argus test
```

The wrappers do not run loading, functionality generation, embedding, splitting, clustering search, encoding inspection, or classification search.

## Experiment checkpoints

Do not move to the next stage until the current checkpoint passes:

| Checkpoint               | Required evidence                                           |
| ------------------------ | ----------------------------------------------------------- |
| ingestion                | repository, file, and snippet counts are non-zero           |
| functionality generation | processed repositories and populated functionality tags     |
| embedding                | Qdrant collection exists with expected point count          |
| splitting                | PostgreSQL and Qdrant both contain train/test assignments   |
| K selection              | candidate report, quality plot, rationale, and accepted K   |
| clustering               | trained model, full quality report, and ids for both splits |
| encoding                 | inspected distribution and ordered sequence are consistent  |
| classifier search        | selected validation result and recorded hyperparameters     |
| classifier train         | best checkpoint and train metrics                           |
| final test               | test predictions and metrics from the frozen checkpoint     |

## Reproducibility record

Archive the following for each reported experiment:

- source revision
- all YAML configuration files
- dependency lock file
- dataset and Qdrant collection identifiers
- split counts and seeds
- clustering candidate and final quality reports
- accepted K and written rationale
- classifier search summary
- final checkpoint, predictions, and metrics
- runtime hardware and elapsed times
