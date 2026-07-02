# Functionality Clustering

This stage maps functionality embeddings into cluster ids. The cluster ids become the semantic vocabulary used for repository classification.

For the step-by-step experimental workflow and interpretation checklist, see [08_CLUSTERING_RESEARCH_GUIDE.md](./08_CLUSTERING_RESEARCH_GUIDE.md).

## CLI entrypoint

```bash
sastllm cluster --mode search
sastllm cluster --mode train
sastllm cluster --mode test
```

Implementation paths:

- Pipeline wrapper: `src/scripts/pipelines.py::cluster_functionalities`
- Stage service: `src/sastllm/clustering/service.py::FunctionalityClusteringService`
- Config parser: `src/sastllm/clustering/config.py`
- Qdrant source adapter: `src/sastllm/clustering/sources.py`
- Clustering model: `src/sastllm/clustering/kmeans.py::MiniBatchKMeansClusterer`
- Embedding source: `src/sastllm/db/managers/embeddings_manager.py`

## Input

Clustering reads functionality vectors from Qdrant. The collection name comes from `configs/split.yaml`:

```text
sentence-transformers_all-mpnet-base-v2
```

The train/test source is filtered by Qdrant payload:

| Mode | Qdrant payload filter |
| --- | --- |
| `train` | `split == "train"` |
| `test` | `split == "test"` |
| `search` | no split filter; reads first `n` embeddings |

Each yielded item is:

```text
(functionality_id, embedding_vector)
```

## Configuration

Current `configs/clustering.yaml`:

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
    load_model_file: "models/clustering/trained_models/clusterer_k_10661.joblib"
```

## Model

`MiniBatchKMeansClusterer` uses `sklearn.cluster.MiniBatchKMeans`.

Important behavior:

- embeddings are L2-normalized before fitting and prediction
- fitting streams vectors in batches of `1000`
- candidate and train models are initialized from the seeded reservoir when it contains at least `k` rows, avoiding dependence on Qdrant stream order
- model state is saved with `joblib`
- saved payload contains `minibatch_kmeans_model` and `n_clusters`

## Search mode

`search` estimates and validates an appropriate `k`. Candidate fitting remains streaming, while every candidate is evaluated against the same seeded reservoir sample so metrics are directly comparable.

For each value `n` in `search.grid_search`:

1. Read up to `n` embeddings from Qdrant.
2. Build one seeded reservoir sample of up to `evaluation.sample_size` normalized embeddings.
3. Compute `k_max = n // m_min`, where `m_min` defaults to `20`.
4. Build logarithmically spaced candidate values between `2` and `k_max`.
5. Fit `MiniBatchKMeans` for each candidate.
6. Evaluate normalized inertia, cohesion, separation, Calinski-Harabasz, cluster-stratified sampled silhouette, and cluster-size health on the fixed sample.
7. Detect the normalized-inertia elbow.
8. Within the configured elbow neighborhood, prefer the best reliable silhouette score.
9. Fall back to Calinski-Harabasz, then the elbow, if silhouette is unavailable or dominated by singleton sampled clusters.
10. Persist candidate metrics, the selection rationale, and a six-panel quality plot.
11. Refit the selected K and run a full streaming quality evaluation.
12. Save the selected model and final full quality report.

The previous fallback of selecting the lowest-inertia candidate was removed because inertia decreases monotonically and therefore biased selection toward the largest K.

## Quality metrics

The evaluator reports complementary evidence rather than hiding it in one weighted score:

| Metric | Meaning | Direction |
| --- | --- | --- |
| normalized inertia | mean squared point-to-assigned-centroid distance | lower is better |
| cohesion RMS | root mean squared point-to-centroid distance | lower is better |
| separation RMS | weighted centroid dispersion around the global centroid | higher is better |
| Calinski-Harabasz | between-cluster dispersion relative to within-cluster dispersion | higher is better |
| silhouette | sampled cohesion versus nearest-cluster separation | higher is better; range `[-1, 1]` |

Euclidean silhouette is used because the current MiniBatchKMeans objective is Euclidean. Embeddings are L2-normalized before both fitting and evaluation.

Full silhouette is quadratic and infeasible for approximately 1.9 million embeddings. The evaluator therefore chooses clusters from the fixed reservoir and samples up to an equal per-cluster budget, targeting at least `silhouette_samples_per_cluster` rows for each chosen cluster. This prevents a small uniform sample from degenerating into thousands of singleton labels when `k` is large.

The report includes silhouette sample size, represented cluster count, singleton sampled clusters, and cluster coverage. Automatic selection checks fragmentation on the broader evaluation reservoir: silhouette is rejected when the fraction of represented clusters containing only one reservoir row exceeds `max_silhouette_singleton_fraction`. `evaluation.sample_size` should be comfortably larger than the expected K; the current `100000` rows provide both a useful quality sample and order-independent initialization for `k=10661`.

Cluster health includes empty, singleton, and below-minimum cluster counts; size percentiles; and per-cluster inertia/RMS radius. Search health values describe the evaluation sample. The selected-model report computes cluster sizes and cohesion over the full streamed source.

Search model filename:

```text
models/clustering/searching_models/clusterers_<n>_<timestamp>/clusterer_<n>_<k>_<timestamp>/clusterer_<n>_<k>_<timestamp>.joblib
```

All search artifacts from one population run share the same directory and timestamp:

```text
models/clustering/searching_models/clusterers_<n>_<timestamp>/
  clusterer_<n>_<k>_<timestamp>/
    clusterer_<n>_<k>_<timestamp>.joblib
    clusterer_<n>_<k>_<timestamp>_quality.json
    clusterer_<n>_<k>_<timestamp>_selection_quality.json
    clusterer_<n>_<k>_<timestamp>_selection_candidates.csv
    clusterer_<n>_<k>_<timestamp>_selection_quality.png
```

## Train mode

`train` fits the configured `k` on train-split embeddings:

1. Read embeddings with payload `split=train`.
2. Build a seeded reservoir of at least `k` rows and use it to initialize `MiniBatchKMeans` without stream-order bias.
3. Fit `MiniBatchKMeans(k=10661)` over the complete stream.
4. Predict cluster ids for the same train embeddings.
5. Save the trained model:

    ```text
    models/clustering/trained_models/clusterer_10661_<timestamp>/clusterer_10661_<timestamp>.joblib
    ```

6. Run a full streaming quality evaluation and save `clusterer_<k>_<timestamp>_quality.json` beside the model.
7. Bulk-update `functionalities.cluster_id` in PostgreSQL.

The update maps:

```text
functionality_id -> cluster_id
```

## Test mode

`test` loads the configured trained model and assigns clusters to test-split embeddings:

1. Load the timestamped model configured in `test.load_model_file`.
2. Read embeddings with payload `split=test`.
3. Predict cluster ids.
4. Bulk-update `functionalities.cluster_id`.

No new model is trained in test mode.

## Output

| Destination | Content |
| --- | --- |
| `functionalities.cluster_id` | integer cluster id assigned to each functionality |
| `models/clustering/.../*.joblib` | trained MiniBatchKMeans model |
| `models/clustering/.../*_quality.json` | full selected/final quality reports and per-cluster statistics |
| `models/clustering/searching_models/clusterers_<n>_<timestamp>/clusterer_<n>_<k>_<timestamp>/` | search model, candidate metrics, selection rationale, CSV, and quality plot |
| `models/clustering/trained_models/clusterer_<k>_<timestamp>/` | trained model and full quality report |

## Downstream contract

Repository classification assumes:

- `repositories.split` is assigned
- repositories are marked `processed=true`
- each included functionality has a valid `cluster_id`
- classifier `k` equals the clustering `k`

If classifier `k` is smaller than the maximum assigned cluster id, vector encoding will fail or silently lose representational consistency depending on the path used.
