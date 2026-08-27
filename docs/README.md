# Documentation Index

This directory contains the reference documentation for the SAST-LLM thesis project.

## Start here

- [PIPELINE.md](./PIPELINE.md) gives the end-to-end thesis pipeline overview.
- [09_FULL_PIPELINE_RUNBOOK.md](./09_FULL_PIPELINE_RUNBOOK.md) gives the exact command sequence and experiment checkpoints.
- [USAGE.md](./USAGE.md) lists CLI commands and typical execution flows.
- [SETUP.md](./SETUP.md) describes local installation, services, and environment variables.
- [CONFIG.md](./CONFIG.md) explains the YAML configuration files.
- [DB_SCHEMA.md](./DB_SCHEMA.md) describes PostgreSQL tables, ORM models, and trigger behavior.

## Pipeline stage references

- [01_DATA_INGESTION_AND_CHUNKING.md](./01_DATA_INGESTION_AND_CHUNKING.md)
- [02_FUNCTIONALITY_GENERATION.md](./02_FUNCTIONALITY_GENERATION.md)
- [03_EMBEDDING_AND_SPLITTING.md](./03_EMBEDDING_AND_SPLITTING.md)
- [04_FUNCTIONALITY_CLUSTERING.md](./04_FUNCTIONALITY_CLUSTERING.md)
- [05_REPOSITORY_CLASSIFICATION.md](./05_REPOSITORY_CLASSIFICATION.md)
- [06_STORAGE_AND_DATA_ACCESS.md](./06_STORAGE_AND_DATA_ACCESS.md)
- [07_CONFIGURATION_REFERENCE.md](./07_CONFIGURATION_REFERENCE.md)
- [08_CLUSTERING_RESEARCH_GUIDE.md](./08_CLUSTERING_RESEARCH_GUIDE.md) provides the clustering experiment runbook and metric interpretation checklist.
- [09_FULL_PIPELINE_RUNBOOK.md](./09_FULL_PIPELINE_RUNBOOK.md) provides the full raw-data-to-test-metrics execution guide.

## Current implementation note

The implemented command pipeline is database-backed and uses PostgreSQL for repositories, files, snippets, and functionalities. Embeddings are stored in Qdrant. The high-level `train` and `test` commands currently run clustering plus classification; they do not run loading, splitting, or LLM generation automatically.
