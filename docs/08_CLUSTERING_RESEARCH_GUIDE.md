# Clustering Research Guide

This guide describes the recommended experimental workflow for selecting, training, validating, and applying functionality clusters. It is the practical companion to [04_FUNCTIONALITY_CLUSTERING.md](./04_FUNCTIONALITY_CLUSTERING.md).

## Research objective

The clustering stage converts functionality-tag embeddings into a semantic vocabulary:

```text
functionality tag
  -> normalized embedding
  -> functionality cluster id
  -> repository encoding
  -> repository classifier
```

The experiment must answer two questions:

1. Are members of each cluster cohesive?
2. Are different clusters sufficiently separated?

No single metric proves that a K is correct. The pipeline therefore reports cohesion, separation, silhouette, Calinski-Harabasz, inertia, and cluster-size health together.

## Mode summary

| Mode | Input | Purpose | Main side effects |
| --- | --- | --- | --- |
| `search` | first `n` vectors from the Qdrant collection | compare candidate K values and select one | writes candidate reports, plots, selected model, and final quality report |
| `train` | Qdrant vectors with `split=train` | fit the chosen K and validate it on the full train stream | writes the trained model and train quality report; updates train functionality cluster ids in PostgreSQL |
| `test` | Qdrant vectors with `split=test` | apply the frozen trained model | updates test functionality cluster ids in PostgreSQL |

`test` does not fit a model or choose K.

## Prerequisites

Before clustering:

- PostgreSQL and Qdrant must be running.
- Functionality records and tags must exist in PostgreSQL.
- Every functionality being clustered must have an embedding in Qdrant.
- Qdrant points must have `split=train` or `split=test` payloads.
- `configs/split.yaml` must name the embedding model used to create the collection.

The collection name is derived from `split.model_name` by replacing `/` with `_`. For example:

```text
sentence-transformers/all-mpnet-base-v2
  -> sentence-transformers_all-mpnet-base-v2
```

Important: `uv run sastllm split` currently assigns split metadata but does not create missing embeddings because `embed_all_repositories()` is disabled in the command wrapper.

Start and check the services:

```bash
docker compose up -d
docker compose ps
uv run sastllm cluster --help
```

## Step 1: Configure the search

Edit `configs/clustering.yaml`:

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
```

Parameter guidance:

| Parameter | Research meaning |
| --- | --- |
| `grid_search` | population sizes to investigate; each value starts an independent search |
| `min_samples_per_cluster` | limits the largest candidate to `n // min_samples_per_cluster` |
| `num_k_candidates` | number of logarithmically spaced K values |
| `sample_size` | fixed reservoir used to compare candidates and initialize models |
| `silhouette_sample_size` | maximum rows used in the quadratic silhouette calculation |
| `silhouette_samples_per_cluster` | target minimum representation of each selected silhouette cluster |
| `elbow_window_factor` | limits automatic silhouette/CH selection to the inertia-elbow neighborhood |
| `max_silhouette_singleton_fraction` | rejects silhouette evidence from an excessively fragmented reservoir |

Keep `sample_size` comfortably above the expected K. The current `100000` supports the expected range around `K=10661`, while keeping the full dataset out of memory.

## Step 2: Run K search

```bash
uv run sastllm cluster --mode search
```

For each `n`, the command:

1. Creates one seeded uniform reservoir of normalized embeddings.
2. Generates logarithmically spaced candidate K values.
3. Fits a streaming MiniBatchKMeans model for every candidate.
4. Evaluates every candidate against the same reservoir.
5. Detects an inertia elbow.
6. Selects the best reliable silhouette result near that elbow.
7. Falls back to Calinski-Harabasz and then the elbow when silhouette is unreliable.
8. Refits the selected K and evaluates cohesion and cluster sizes over the full stream.

Search may be expensive. With 30 candidates and approximately 1.9 million embeddings, the source is scanned repeatedly and each candidate stores K centroids. Progress is shown as `Evaluating candidate K`.

### Search outputs

```text
models/clustering/searching_models/clusterers_<n>_<timestamp>/
  clusterer_<n>_<selected_k>_<timestamp>/
    clusterer_<n>_<selected_k>_<timestamp>.joblib
    clusterer_<n>_<selected_k>_<timestamp>_quality.json
    clusterer_<n>_<selected_k>_<timestamp>_selection_quality.json
    clusterer_<n>_<selected_k>_<timestamp>_selection_candidates.csv
    clusterer_<n>_<selected_k>_<timestamp>_selection_quality.png
```

The plot contains:

- normalized inertia
- sampled silhouette
- Calinski-Harabasz
- cohesion RMS
- separation RMS
- silhouette cluster coverage

The JSON report records `selected_k`, `selection_reason`, `elbow_k`, `silhouette_best_k`, `calinski_harabasz_best_k`, and all candidate measurements.

## Step 3: Review the selected K

Treat automatic selection as a recommendation, not as the thesis conclusion. Review the CSV, JSON, and plot together.

| Evidence | Desired behavior | Warning sign |
| --- | --- | --- |
| cohesion RMS | low relative to nearby candidates | little improvement despite rapidly increasing K |
| separation RMS | high relative to nearby candidates | separation remains flat while K grows |
| silhouette | highest reliable value near the elbow | negative/near-zero score or very low coverage |
| Calinski-Harabasz | local maximum near selected K | best value occurs far from the elbow/silhouette result |
| normalized inertia | clear diminishing-return elbow | choosing the largest K merely because inertia is lowest |
| cluster sizes | few empty, singleton, or undersized clusters | many tiny clusters, indicating fragmentation |

Silhouette interpretation is contextual:

- close to `1`: compact, clearly separated sampled clusters
- around `0`: overlapping or boundary-heavy clusters
- below `0`: many sampled points are closer to another cluster

There is no universal acceptance threshold for semantic embeddings. Report the value, sampling method, coverage, competing K values, and cluster-size distribution instead of claiming that one score proves validity.

For the selected model, prefer the `_quality.json` beside the model when reporting full cohesion and cluster-size statistics. Candidate reports are sample-based; the selected-model report has `scope: "full"` and streams all evaluated vectors for non-silhouette metrics.

## Step 4: Synchronize K

After accepting a K, update all dependent configuration fields.

In `configs/clustering.yaml`:

```yaml
clustering:
  train:
    k: <selected_k>
  test:
    k: <selected_k>
    load_model_file: "models/clustering/trained_models/clusterer_<selected_k>_<timestamp>/clusterer_<selected_k>_<timestamp>.joblib"
```

In `configs/classification.yaml`, set `params.k` under every classification mode that will be used:

```yaml
classification:
  search:
    params:
      k: <selected_k>
  train:
    params:
      k: <selected_k>
  test:
    params:
      k: <selected_k>
```

The classifier feature dimension or token vocabulary depends on K. A mismatch can invalidate repository encodings or make a saved classifier incompatible.

## Step 5: Train the final clusterer

```bash
uv run sastllm cluster --mode train
```

Expected behavior:

1. Count train embeddings.
2. Fail clearly if fewer than K train embeddings exist.
3. Build a seeded reservoir containing at least K rows.
4. Initialize MiniBatchKMeans from that reservoir to avoid Qdrant stream-order bias.
5. Fit over all `split=train` vectors.
6. Save the model and full train quality report.
7. Assign zero-based cluster ids to train functionalities in PostgreSQL.

Outputs:

```text
models/clustering/trained_models/clusterer_<k>_<timestamp>/
  clusterer_<k>_<timestamp>.joblib
  clusterer_<k>_<timestamp>_quality.json
```

Inspect the train quality report before classification. Confirm that:

- `evaluated_samples` matches the expected train functionality count
- `represented_clusters` is close to K
- `empty_clusters`, `singleton_clusters`, and `below_minimum_clusters` are acceptable
- full-stream cohesion agrees reasonably with search estimates
- sampled silhouette remains consistent with the selected candidate

## Step 6: Apply the model to test data

Confirm that `test.load_model_file` points to the newly trained artifact, then run:

```bash
uv run sastllm cluster --mode test
```

Expected behavior:

- the model is loaded without refitting
- only `split=test` vectors are read
- cluster ids are written for test functionalities
- the training vocabulary and centroid positions remain frozen

Test mode currently writes assignments but does not produce a separate test quality report.

## Step 7: Continue to classification

Run classification explicitly so clustering results can be inspected between stages:

```bash
uv run sastllm classify --mode train
uv run sastllm classify --mode test
```

The wrappers are available after configuration is finalized:

```bash
uv run sastllm train
uv run sastllm test
```

They execute:

```text
train -> cluster train -> classification train
test  -> cluster test  -> classification test
```

They do not perform data loading, functionality generation, embedding creation, splitting, or K search.

## Methodological caveat

The current `search` implementation reads the first `n` vectors from the complete Qdrant collection without filtering on `split=train`. This is a transductive vocabulary-selection setup: test embeddings can influence K even though labels are not used.

For an explicitly inductive evaluation, K selection should be changed to use only `split=train`. Until then, state the search scope in the thesis and do not describe K as selected exclusively from training data.

## Reproducibility checklist

Record these items for every experiment:

- Git commit or source revision
- full `clustering.yaml` and `split.yaml`
- Qdrant collection name and embedding model
- population and train/test functionality counts
- random seeds
- candidate CSV and quality plot
- selection JSON and stated rationale
- selected model quality JSON
- final K used by classification
- software environment and hardware

Running search again with the same `n` and output directories overwrites files with the same names. Use a separate experiment directory or archive each run before comparing configurations.

## Recommended command sequence

```bash
# Services and preconditions
docker compose up -d
uv run sastllm split

# K selection and review
uv run sastllm cluster --mode search

# After synchronizing the accepted K in both YAML files
uv run sastllm cluster --mode train
uv run sastllm cluster --mode test

# Downstream experiment
uv run sastllm classify --mode train
uv run sastllm classify --mode test
```

Do not run `split` unless the embeddings already exist and PostgreSQL/Qdrant are ready to receive synchronized split metadata.
