# Functionality Generation

This stage turns code snippets into normalized natural-language functionality tags.

## CLI entrypoints

Online LLM generation:

```bash
sastllm generate_functionalities
```

OpenAI Batch API generation:

```bash
sastllm generate_functionalities_batch_api
```

Cached functionality import:

```bash
sastllm load_cache_functionalities /path/to/cached_functionalities
```

Implementation paths:

- Online processor: `src/sastllm/processors/snippet_processor.py`
- Prompt formatter: `src/sastllm/prompt/functionality_prompt_generator.py`
- LLM analyzer: `src/sastllm/analyzers/functionality_analyzer.py`
- Batch file generator: `src/sastllm/processors/batch_files_generator.py`
- Batch submitter/downloader: `src/sastllm/processors/batch_file_processor.py`
- LLM factory: `src/scripts/utils.py::get_model`

## Online generation flow

`generate_functionalities()` creates the configured `snippet_processor` LLM, then runs `SnippetProcessor` with:

| Setting | Current value |
| --- | --- |
| Batch size | `50` snippets |
| Sleep between batches | `5` seconds |
| Retry attempts for LLM call | `1000` |
| Initial retry delay | `1` second |

The flow is:

1. Fetch unprocessed snippets with file metadata from PostgreSQL.
2. Build a contextual prompt for a batch of snippets.
3. Ask the configured LLM to produce functionality lines.
4. Parse the LLM output.
5. Deduplicate descriptions per snippet.
6. Normalize every description into a tag.
7. Insert `functionalities` rows.
8. Mark the snippet as processed.
9. Cache the parsed functionality JSON under `cache/functionalities-<llm_type>/`.

## Prompt structure

Before a snippet is sent to the LLM, `FunctionalityPromptGenerator` enriches it with:

- snippet id
- file path
- detected function name, when available
- language
- AST-derived function calls
- AST-derived control structures
- source code, truncated to `400` lines if needed

The LLM receives rules that ask it to:

- summarize behavior as semantic, goal-oriented actions
- avoid syntax-level details such as registers, variables, opcodes, or implementation minutiae
- use short imperative sentences
- separate multiple functionalities with semicolons
- output one line per snippet/chunk

Expected output format:

```text
<snippet_id>: <functionality 1>; <functionality 2>; <functionality 3>
```

Example:

```text
17: Load configuration file; Decode encoded payload; Execute generated command
18: None
```

## Parsing and normalization

The parser accepts lines with a numeric id followed by `:`. Functionalities are split on `;`.

Special handling:

- blank lines are ignored
- malformed lines are skipped
- a single functionality equal to `None` becomes an empty list
- duplicate descriptions for the same snippet are removed while preserving order

Each description is stored twice:

- `description`: raw LLM functionality sentence
- `tag`: normalized text from `sastllm.utils.Normalizer`

The tag is what later stages embed and cluster.

## Batch API path

The batch command currently:

1. Uses model `gpt-5-mini`.
2. Writes request files under `api_batches_extra/`.
3. Uses `BatchFilesGenerator` with `snippet_batch_size=20` and `api_batch_size=500`.
4. Submits JSONL files to OpenAI Batch API with endpoint `/v1/responses`.
5. Polls every `60` seconds for up to `30` hours.
6. Downloads output/error JSONL files under `batch_results_extra/`.

Important limitation: the batch submitter downloads raw result files, but does not itself parse those downloaded Batch API outputs back into the database. Database import is handled by `load_cache_functionalities`, which expects local JSON files named like:

```text
functionalities_<snippet_id>.json
```

Each JSON file should contain objects compatible with `CreateFunctionalityDto`.

## LLM configuration

`configs/llms.yaml` currently defines only the `snippet_processor` model:

```yaml
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

Supported hosts in the factory are:

| Host | Implementation |
| --- | --- |
| `google` | `ChatGoogleGenerativeAI` |
| `openai` | `OpenAI` from `langchain_openai` |
| `issel` | local `CustomLLM` using `ENDPOINT_URL` and `ACCESS_TOKEN` |

## Outputs

| Destination | Content |
| --- | --- |
| `functionalities.description` | raw functionality sentence |
| `functionalities.tag` | normalized text used for embeddings |
| `functionalities.cluster_id` | initially `NULL` |
| `snippets.processed` | set to `true` after processing |
| `cache/functionalities-<llm_type>/` | debugging/import cache |

## Failure behavior

- A failed online LLM call is retried with exponential backoff.
- If a batch fails after retries, processing stops at that batch.
- Bulk functionality insert falls back to single-row inserts.
- If a snippet yields no functionality, the snippet can still be marked processed.
