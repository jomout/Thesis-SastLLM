import os
import re
from typing import Dict, List, Tuple

import chardet
import yaml
from tqdm import tqdm

from sastllm.configs import get_logger
from sastllm.db import FileManager, RepositoryManager, SnippetManager
from sastllm.dtos import CreateRepositoryDto
from sastllm.dtos.create_dtos import CreateFileDto, CreateSnippetDto
from sastllm.parsers import CodeChunker

logger = get_logger(__name__)


def _normalize_code(src: str) -> str:
    s = src.replace("\r\n", "\n").replace("\r", "\n")
    if not s.endswith("\n"):
        s += "\n"
    return s


class CodeProcessor:
    """
    Recursively loads source code files from a directory, parses them,
    and yields them as CodeSnippet objects after chunking.

    This class enables processing large source code repositories by breaking files
    into manageable chunks for downstream tasks such as machine learning or static analysis.
    """

    def __init__(
        self,
        *,
        root_path: str,
        max_tokens: int = 100,
    ) -> None:
        """
        Initializes the CodeLoader.

        Args:
            root_path (str):
                The root directory to recursively search for code files.
            max_tokens (int):
                Maximum number of tokens per chunk. Defaults to 100.
        """
        logger.debug("Initializing CodeProcessor.")

        self.root_path = root_path
        self.max_tokens = max_tokens

        self.repository_db = RepositoryManager()
        self.file_db = FileManager()
        self.snippet_db = SnippetManager()

        self._repo_cache: Dict[str, int] = {}  # repo_name -> repository_id

        self.extensions_to_language = self._load_extension_mapping_from_yaml("configs/languages.yaml")

        self.suffixes = list(self.extensions_to_language.keys())

        logger.debug("CodeProcessor initialized.")

    def _bootstrap_repo_cache(self) -> None:
        """
        Loads all repositories from the database into an in-memory cache for quick lookup.
        """
        logger.debug("Bootstrapping repository cache from DB.")
        self._repo_cache = {r.name: r.repository_id for r in self.repository_db.get_repositories()}
        logger.debug(f"Loaded {len(self._repo_cache)} repositories into cache.")

    def _get_or_create_repository(self, repo_name: str, label: str) -> int:
        """
        Return repository_id for repo_name; create row if absent.
        """
        rid = self._repo_cache.get(repo_name)
        if rid is not None:
            return rid
        logger.debug(f"Creating repository row for: {repo_name}")

        repository = CreateRepositoryDto(name=repo_name, label=label)
        rid = self.repository_db.add_repository(repository)
        self._repo_cache[repo_name] = rid
        return rid

    def _infer_repo_name_and_label(self, rel_path: str) -> Tuple[str, str]:
        """
        Given a relative path, return (repo_name, label).
        - label = top-level folder under root (apt, rat, worm, etc.)
        - repo_name = second-level folder (e.g. AsyncRAT, APT34).
        If a file lives directly under the label folder, repo_name = that label.
        """
        parts = rel_path.split(os.sep)
        if len(parts) >= 2:
            label = parts[0]
            repo_name = parts[1]
        else:
            # fallback for files directly in root
            label = os.path.basename(os.path.abspath(self.root_path))
            repo_name = label
        return repo_name, label

    def _get_files(self) -> List[str]:
        """
        Recursively collects files from the root path that match the specified suffixes.

        Returns:
            List[str]:
                List of file paths matching the given suffixes.
        """
        logger.debug("Filtering files to match given suffixes")

        matched_files = []
        for dirpath, _, filenames in os.walk(self.root_path):
            for filename in filenames:
                # Skip resource fork files (e.g., '._file.py')
                if filename.startswith("._"):
                    continue
                if any(filename.endswith(f".{suffix}") for suffix in self.suffixes):
                    matched_files.append(os.path.join(dirpath, filename))

        logger.debug("Filtering files completed.")

        return matched_files

    def parse_file(self, file_path: str) -> str:
        """
        Reads a source code file, decodes safely, removes NUL bytes,
        and strips non-ASCII characters from string literals.
        """
        logger.debug(f"Parsing file: {file_path}")

        with open(file_path, "rb") as f:
            raw = f.read()

        # Try UTF-8 first, fall back if necessary
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError:
            detected = chardet.detect(raw)["encoding"] or "latin-1"
            logger.warning(f"Non-UTF8 file detected ({file_path}), decoding with {detected}")
            source = raw.decode(detected, errors="replace")

        # Remove NULs (critical for Postgres!)
        source = source.replace("\x00", "")

        # Strip non-ASCII chars inside string literals
        def strip_non_ascii_chars(match):
            s = match.group(0)
            quote = s[0]
            inner = s[1:-1]
            cleaned = "".join(ch for ch in inner if ch.isascii())
            return f"{quote}{cleaned}{quote}"

        source = re.sub(r'(["\'])(?:(?=(\\?))\2.)*?\1', strip_non_ascii_chars, source)

        logger.debug(f"Parsed and cleaned file: {file_path}")
        return source

    def run(self) -> None:
        """
        Processes all code files from the root path, chunks them, and stores the chunks in the
        database.
        Steps:
            1. Recursively fetch all files from the root directory.
            2. For each file:
                a. Parse the file to extract source code.
                b. Normalize the code formatting.
                c. Chunk the code into smaller pieces based on max_tokens.
                d. Store each chunk in the database with metadata.
        """
        logger.info(f"Starting code processing from root path: {self.root_path}")

        self._bootstrap_repo_cache()

        chunker = CodeChunker(max_tokens=self.max_tokens, remove_comments=True)

        for file_path in tqdm(self._get_files(), desc="Parsing files", leave=True):
            ext = os.path.splitext(file_path)[1].lower().lstrip(".")
            rel_path = os.path.relpath(file_path, self.root_path)

            # Fetch repository name and label
            repo_name, label = self._infer_repo_name_and_label(rel_path)
            filename = os.path.basename(file_path)

            source_code = self.parse_file(file_path)
            normalized_code = _normalize_code(source_code)

            language = self.extensions_to_language.get(ext)
            if not language:
                logger.warning(f"Skipping unsupported file extension: .{ext} (file: {file_path})")
                continue

            try:
                chunks = chunker.chunk_code(code=normalized_code, language=language)
            except Exception as e:
                logger.warning(f"Failed to chunk file {file_path}: {e}")
                continue

            # Add repository
            repository_id = self._get_or_create_repository(repo_name=repo_name, label=label)

            # Add file to database
            file = CreateFileDto(
                repository_id=repository_id,
                language=language,
                filename=filename,
                filepath=rel_path,
            )
            file_id = self.file_db.add_file(file)

            # Add all snippets to database in bulk (fallback to per-snippet on failure)
            try:
                snippets = [
                    CreateSnippetDto(
                        file_id=file_id,
                        code=c["code"],
                        start_line=c["start_line"],
                        end_line=c["end_line"],
                    )
                    for c in chunks.values()
                ]
                if snippets:
                    _ = self.snippet_db.add_bulk_snippets(snippets)
            except Exception as e:
                logger.warning(f"Bulk insert failed for file {file_path}, falling back to single inserts: {e}")
                for chunk in chunks.values():
                    try:
                        snippet = CreateSnippetDto(
                            file_id=file_id,
                            code=chunk["code"],
                            start_line=chunk["start_line"],
                            end_line=chunk["end_line"],
                        )
                        _ = self.snippet_db.add_snippet(snippet=snippet)
                    except Exception as e2:
                        logger.warning(f"Failed to add snippet for file {file_path}: {e2}")
                        continue

        logger.info("Code processing completed successfully.")

    @staticmethod
    def _load_extension_mapping_from_yaml(config_path: str) -> Dict[str, str]:
        """
        Loads the language extension mapping from a YAML configuration file.

        Args:
            config_path (str): The file path to the YAML configuration.

        Returns:
            Dict[str, str]: A mapping of file extensions to programming languages.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"languages.yaml not found at '{config_path}'")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        mapping: Dict[str, str] = {}
        for lang in cfg.get("languages") or []:
            name = lang.get("name")
            for suf in lang.get("suffixes") or []:
                if not isinstance(suf, str):
                    continue
                mapping[suf.lower()] = name
        return mapping
