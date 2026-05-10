# Functionality Clustering

This stage maps functionality embeddings into cluster ids. The cluster ids become the semantic vocabulary used for repository classification.

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

## Model

`MiniBatchKMeansClusterer` uses `sklearn.cluster.MiniBatchKMeans`.

Important behavior:

- embeddings are L2-normalized before fitting and prediction
- fitting streams vectors in batches of `1000`
- model state is saved with `joblib`
- saved payload contains `minibatch_kmeans_model` and `n_clusters`

## Search mode

`search` estimates an appropriate `k`.

For each value `n` in `search.grid_search`:

1. Read up to `n` embeddings from Qdrant.
2. Compute `k_max = n // m_min`, where `m_min` defaults to `20`.
3. Build 30 logarithmically spaced candidate values between `2` and `k_max`.
4. Fit `MiniBatchKMeans` for each candidate.
5. Track inertia.
6. Use `KneeLocator` with a convex decreasing curve to detect the elbow.
7. Stop early if the detected knee is stable for several iterations.
8. Save an inertia plot under `save_plots_dir`.
9. Fit and save a model for the selected `k`.

Search model filename:

```text
clusterer_n_<n>_k_<optimal_k>.joblib
```

## Train mode

`train` fits the configured `k` on train-split embeddings:

1. Read embeddings with payload `split=train`.
2. Fit `MiniBatchKMeans(k=10661)`.
3. Predict cluster ids for the same train embeddings.
4. Save the trained model:

    ```text
    models/clustering/trained_models/clusterer_k_10661.joblib
    ```

5. Bulk-update `functionalities.cluster_id` in PostgreSQL.

The update maps:

```text
functionality_id -> cluster_id
```

## Test mode

`test` loads the configured trained model and assigns clusters to test-split embeddings:

1. Load `models/clustering/trained_models/clusterer_k_10661.joblib`.
2. Read embeddings with payload `split=test`.
3. Predict cluster ids.
4. Bulk-update `functionalities.cluster_id`.

No new model is trained in test mode.

## Output

| Destination | Content |
| --- | --- |
| `functionalities.cluster_id` | integer cluster id assigned to each functionality |
| `models/clustering/.../*.joblib` | trained MiniBatchKMeans model |
| `plots/clustering/searching/*.png` | inertia/elbow plots from search mode |

## Downstream contract

Repository classification assumes:

- `repositories.split` is assigned
- repositories are marked `processed=true`
- each included functionality has a valid `cluster_id`
- classifier `k` equals the clustering `k`

If classifier `k` is smaller than the maximum assigned cluster id, vector encoding will fail or silently lose representational consistency depending on the path used.
