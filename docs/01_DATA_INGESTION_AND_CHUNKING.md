# Data Ingestion And Chunking

This stage converts a local repository dataset into database records: repositories, files, and code snippets.

## CLI entrypoint

```bash
argus load
```

Implementation path:

- CLI command: `src/scripts/cli.py`
- Pipeline wrapper: `src/scripts/pipelines.py::load_dataset`
- Main processor: `src/argus/processors/code_processor.py::CodeProcessor`
- Chunker: `src/argus/parsers/code_chunker.py::CodeChunker`
- Parser helpers: `src/argus/parsers/code_parser.py`, `comment_stripper.py`, `tree_sitter_generator.py`

## Input

The dataset root is read from `configs/base.yaml`:

```yaml
paths:
  dataset: ".dataset/thesis_dataset"
```

The loader expects a label/repository folder shape:

```text
.dataset/thesis_dataset/
  benign/
    repo_a/
      source files...
  apt/
    repo_b/
      source files...
  rat/
    repo_c/
      source files...
```

The first path component below the dataset root becomes the repository label. The second path component becomes the repository name. If a file is directly under the dataset root, the loader falls back to using the root folder name for both.

## File discovery

`CodeProcessor` loads supported suffixes from `configs/languages.yaml`. Each suffix maps to an internal Tree-sitter language name. Examples:

| Suffix         | Language     |
| -------------- | ------------ |
| `.py`          | `python`     |
| `.c`, `.h`     | `c`          |
| `.cpp`, `.hpp` | `cpp`        |
| `.js`, `.jsx`  | `javascript` |
| `.ts`          | `typescript` |
| `.tsx`         | `tsx`        |
| `.java`        | `java`       |
| `.rs`          | `rust`       |
| `.go`          | `go`         |
| `.sh`, `.bash` | `bash`       |

Files whose names start with `._` are skipped.

## Source decoding and cleanup

For every supported file, the loader:

- reads bytes from disk
- tries UTF-8 decoding first
- falls back to `chardet` detection, or `latin-1` if detection fails
- removes NUL bytes before database insertion
- strips non-ASCII characters inside string literals
- normalizes line endings to `\n`
- appends a final newline if missing

This cleanup is mainly there to keep PostgreSQL inserts stable and reduce noisy text in later LLM prompts.

## AST-aware chunking

`CodeChunker` chunks source code with a token budget and syntax-aware breakpoints.

Important defaults:

| Setting                     | Value                              |
| --------------------------- | ---------------------------------- |
| Encoding                    | `cl100k_base`                      |
| Token budget                | `100` tokens                       |
| Comment removal during load | enabled via `remove_comments=True` |

The chunker uses Tree-sitter to find important node start lines. Important nodes come from `configs/important_nodes.yaml`; comment/decorator nodes come from `configs/comment_nodes.yaml`.

The algorithm:

1. Parse source code with the language-specific Tree-sitter parser.
2. Collect important node line numbers and comment line numbers.
3. Move breakpoints upward when a contiguous comment block belongs to the following node.
4. Walk the file line by line and count tokens.
5. When the token budget is exceeded, emit a chunk at the nearest previous safe breakpoint.
6. If no safe breakpoint exists yet, force-include the long line.
7. Strip comments if configured.
8. Normalize whitespace by removing blank lines and trimming the chunk.
9. Store non-empty chunks with 1-based `start_line` and `end_line`.

Long base64-like strings are scrubbed before token counting and replaced with a short hash marker:

```text
[ENCODED_BLOB len=<length> sha256=<prefix>]
```

## Database writes

For each file, the loader writes:

- one `repositories` row, created once per repository name
- one `files` row per discovered source file
- many `snippets` rows per chunked file

Snippet insertion uses a bulk insert first and falls back to individual inserts if the bulk write fails.

## Output tables

| Table          | Output                                                 |
| -------------- | ------------------------------------------------------ |
| `repositories` | repository name, label, split, processed flag          |
| `files`        | language, filename, relative filepath, repository link |
| `snippets`     | chunk code, source line range, processed flag          |

## Operational notes

- Re-running `argus load` can create duplicate file/snippet rows unless the database has been cleaned or deduplicated externally.
- Repository identity is based on repository name, not full path.
- `processed` flags are updated by database triggers once snippet processing progresses.
- Language support depends on Tree-sitter grammar availability and the YAML node mappings.
