# Troubleshooting

Common issues and how to fix them.

## Database connection errors

Symptoms:

- `Missing required database configuration` or connection refused.

Checklist:

- Postgres is running: `docker compose ps`
- Environment variables set: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST=localhost`, `POSTGRES_PORT=5433`
- The container exposes 5433 on the host (compose file does), and your firewall allows it.

## LLM authentication errors (for intermediate analysis)

Symptoms:

- Provider-specific errors like `invalid API key` or `permission denied`.

Checklist:

- `GOOGLE_API_KEY` or `OPENAI_API_KEY` exported in your shell.
- `configs/llms.yaml` host matches the key you set (google vs openai).

## spaCy model not found

Symptoms:

- `Failed to initialize NLP model en_core_web_sm`.

Fix:

```zsh
python -m spacy download en_core_web_sm
```

## Clustering errors (HDBSCAN / scikit-learn)

Symptoms:

- ImportError for `HDBSCAN` or `KMeans`.

Fix:

```zsh
pip install scikit-learn hdbscan joblib
```

## No snippets or functionalities

Checklist:

- `paths.database_dir` points to your dataset root.
- Your dataset has files underneath `malware/` and `benignware/`.
- `CodeChunker` `max_tokens` is not too small; consider 400–800.

## Empty ML training data

Possible causes:

- Repository labels missing or set incorrectly during ingestion.
- No clusters assigned yet (run `sastllm train`).

Fix:

- Ensure repositories have `label` set (malware/benignware) during `sastllm setup`.
- Ensure TagProcessor has assigned `cluster_id` to functionalities.
