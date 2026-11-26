import json
from collections.abc import Iterator
from pathlib import Path

from argus.configs import get_logger
from argus.db import FunctionalityManager, SnippetManager
from argus.dtos import CreateFunctionalityDto, UpdateSnippetDto
from argus.utils import Normalizer

logger = get_logger(__name__)

FUNCTIONALITY_CACHE_DIR = Path("cache/functionalities-openai")


class BatchFunctionalityResultImporter:
    """Import successful Responses API batch results into PostgreSQL."""

    def __init__(self, results_dir: str | Path, recursive: bool = False) -> None:
        self.functionality_db = FunctionalityManager()
        self.snippet_db = SnippetManager()
        self.normalizer = Normalizer()

        self.results_dir = Path(results_dir).expanduser().resolve()
        if not self.results_dir.is_dir():
            raise NotADirectoryError(f"Not a valid Batch API results directory: {self.results_dir}")
        self.recursive = recursive

    def _scan_result_files(self) -> list[Path]:
        """Return successful Batch API output files in deterministic order."""
        pattern = "**/*_output.jsonl" if self.recursive else "*_output.jsonl"
        return sorted(path for path in self.results_dir.glob(pattern) if path.is_file())

    @staticmethod
    def _read_jsonl(path: Path) -> Iterator[dict]:
        """Stream valid JSON objects from a JSONL result file."""
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as jsonl_file:
            for line_number, line in enumerate(jsonl_file, start=1):
                if not (stripped_line := line.strip()):
                    continue
                try:
                    yield json.loads(stripped_line)
                except json.JSONDecodeError as error:
                    logger.warning(
                        "Skipping malformed Batch API result line",
                        file=path.name,
                        line_number=line_number,
                        error=str(error),
                    )

    def _store_functionalities(self, extracted_data: dict[str, list[str]]) -> None:
        """Normalize and store extracted functionalities grouped by snippet id."""
        logger.debug("Storing extracted functionalities in the database")

        for snippet_id_text, descriptions in extracted_data.items():
            try:
                snippet_id = int(snippet_id_text)
                unique_descriptions = list(dict.fromkeys(descriptions))
                payload = [
                    CreateFunctionalityDto(
                        snippet_id=snippet_id,
                        description=description,
                        tag=self.normalizer.normalize_text(description),
                        cluster_id=None,
                    )
                    for description in unique_descriptions
                ]

                snippet = self.snippet_db.get_snippet(snippet_id=snippet_id)
                if snippet and snippet.processed:
                    logger.debug("Snippet already processed; skipping", snippet_id=snippet_id)
                    continue

                if payload:
                    self.functionality_db.add_bulk_functionalities(functionalities=payload)
                else:
                    logger.debug("No functionalities extracted", snippet_id=snippet_id)

                self._cache(snippet_id=snippet_id, payload=payload)
                self.snippet_db.update_snippet(snippet=UpdateSnippetDto(snippet_id=snippet_id, processed=True))
            except Exception as error:  # noqa: BLE001 -- malformed external output is skipped at this boundary
                logger.warning(
                    "Skipping malformed functionality output",
                    snippet_id=snippet_id_text,
                    error=str(error),
                )

        logger.debug("Successfully stored extracted functionalities")

    @staticmethod
    def _cache(snippet_id: int, payload: list[CreateFunctionalityDto]) -> None:
        """Cache extracted functionalities as a best-effort debugging artifact."""
        FUNCTIONALITY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = FUNCTIONALITY_CACHE_DIR / f"functionalities_{snippet_id}.json"
        try:
            with file_path.open("w", encoding="utf-8") as cache_file:
                json.dump(
                    [functionality.model_dump() for functionality in payload],
                    cache_file,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.debug("Cached functionalities", snippet_id=snippet_id, path=str(file_path))
        except Exception as error:  # noqa: BLE001 -- caching is best-effort and must not fail processing
            logger.warning(
                "Failed to cache functionalities",
                snippet_id=snippet_id,
                path=str(file_path),
                error=str(error),
            )

    def import_results(self) -> None:
        """Import all successful output files currently present in the results directory."""
        for jsonl_path in self._scan_result_files():
            for result in self._read_jsonl(jsonl_path):
                text = self._extract_output_text(result)
                if not text:
                    logger.warning(
                        "Batch API result contained no output text",
                        file=jsonl_path.name,
                        custom_id=result.get("custom_id"),
                    )
                    continue

                extracted_data = self._parse_functionality_output(text)
                self._store_functionalities(extracted_data=extracted_data)

    @staticmethod
    def _extract_output_text(result: dict) -> str | None:
        """Extract visible text from a successful Responses API batch result."""
        response = result.get("response")
        if not isinstance(response, dict) or response.get("status_code") != 200:
            return None
        body = response.get("body")
        if not isinstance(body, dict):
            return None
        output = body.get("output")
        if not isinstance(output, list):
            return None

        text_parts: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
        return "\n".join(text_parts) or None

    @staticmethod
    def _parse_functionality_output(output: str) -> dict[str, list[str]]:
        """Parse semicolon-separated functionality descriptions keyed by snippet id."""
        result: dict[str, list[str]] = {}
        for line in output.strip().splitlines():
            if not line.strip():
                continue
            try:
                snippet_id_text, functionalities_text = line.split(":", 1)
                snippet_id = str(int(snippet_id_text.strip()))
                functionalities = [functionality.strip() for functionality in functionalities_text.split(";") if functionality.strip()]
                if len(functionalities) == 1 and functionalities[0].lower() == "none":
                    functionalities = []
                result[snippet_id] = functionalities
            except ValueError:
                continue
        return result
