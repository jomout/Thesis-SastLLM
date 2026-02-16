import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Union

from sastllm.configs import get_logger
from sastllm.db import FunctionalityManager, SnippetManager
from sastllm.dtos import CreateFunctionalityDto, UpdateSnippetDto
from sastllm.utils import Normalizer

logger = get_logger(__name__)

class JsonlFolder:
    """
    Utility class to manage and iterate over all .jsonl files in a folder.
    """

    def __init__(self, folder: Union[str, Path], recursive: bool = False):

        self.functionality_db = FunctionalityManager()
        self.snippet_db = SnippetManager()

        self.normalizer = Normalizer()

        self.folder = Path(folder).expanduser().resolve()
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Not a valid folder: {self.folder}")
        self.recursive = recursive
        self._jsonl_files: List[Path] = self._scan()

    def _scan(self) -> List[Path]:
        """Scan the folder for .jsonl files (recursively if enabled)."""
        pattern = "**/*.jsonl" if self.recursive else "*.jsonl"
        files = sorted(self.folder.glob(pattern))
        return [f for f in files if f.is_file()]

    def refresh(self) -> None:
        """Rescan the folder for updated contents."""
        self._jsonl_files = self._scan()

    def _read_jsonl(self, path: Union[str, Path]) -> Iterator[Dict]:
        """Stream JSON objects from a given .jsonl file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"# Warning: {path.name}: {e}")
                    continue

    def _store_functionalities(self, extracted_data: Dict[str, List[str]]) -> None:
        """
        Normalizes and stores extracted functionalities in the database.
        
        Args:
            extracted_data (Dict[str, List[str]]): Parsed functionalities grouped by chunk.
        """
        logger.debug("Storing extracted functionalities in the database.")
        
        for key, descriptions in extracted_data.items():
            try:
                # Extract the numeric snippet ID from the key
                snippet_id = int(key)

                # Load unique descriptions only
                unique_descriptions = list(dict.fromkeys(descriptions))

                # Prepare bulk payload
                payload = [
                    CreateFunctionalityDto(
                        snippet_id=snippet_id,
                        description=desc,
                        tag=self.normalizer.normalize_text(desc),
                        cluster_id=None,
                    )
                    for desc in unique_descriptions
                ]
                
                # Mark snippet as processed
                _snippet = self.snippet_db.get_snippet(snippet_id=snippet_id)
                if _snippet and _snippet.processed:
                    logger.debug(f"Snippet {snippet_id} already processed. Skipping.")
                    continue
                snippet = UpdateSnippetDto(snippet_id=snippet_id, processed=True)
                self.snippet_db.update_snippet(snippet=snippet)

                # Write to file for debugging - caching
                self._cache(dir="cache/functionalities-openai", snippet_id=snippet_id, payload=payload)

                if not payload:
                    logger.debug(f"No functionalities extracted for {snippet_id}. Skipping.")
                    continue

                try:
                    _ = self.functionality_db.add_bulk_functionalities(functionalities=payload)
                except Exception as e:
                    logger.warning(f"Bulk functionality insert failed for {snippet_id}, falling back to single inserts: {e}")
                    for item in payload:
                        try:
                            functionality = CreateFunctionalityDto(
                                snippet_id=item.snippet_id,
                                description=item.description,
                                tag=item.tag,
                                cluster_id=item.cluster_id,
                            )
                            _ = self.functionality_db.add_functionality(functionality=functionality)
                        except Exception as e2:
                            logger.warning(f"Failed to add functionality for {snippet_id}: {e2}")
                            continue
            except Exception as e:
                logger.warning(f"Skipping malformed output for {key}: {e}")
                return
        
        logger.debug("Successfully stored extracted functionalities.")

    @staticmethod
    def _cache(dir: str, snippet_id: int, payload: List[CreateFunctionalityDto]) -> None:
        """
        Caches the extracted functionalities to a JSON file for debugging purposes.
        """
        os.makedirs(dir, exist_ok=True)

        file_path = os.path.join(dir, f"functionalities_{snippet_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([p.model_dump() for p in payload], f, ensure_ascii=False, indent=2)
            logger.debug(f"Wrote functionalities for {snippet_id} to {file_path}")
        except Exception as file_err:
            logger.warning(f"Failed to write file for {snippet_id}: {file_err}")
        


    def read_folder(self) -> None:
        for jsonl_path in self._jsonl_files:
            jsonl_file = self._read_jsonl(jsonl_path)
            for json_obj in jsonl_file:
                try:
                    text = json_obj["response"]["body"]["output"][1]["content"][0]["text"]
                except KeyError as e:
                    logger.warning(f"Malformed JSON object in {jsonl_path.name}, {json_obj}: missing key {e}")
                    break
                if not text:
                    continue

                result = self._parse_functionality_output(text)
                
                self._store_functionalities(extracted_data=result)

    @staticmethod
    def _parse_functionality_output(output: str) -> Dict[str, List[str]]:
        """
        Parse LLM output in the format:
        <chunk_number>: <functionality 1>; <functionality 2>; <functionality 3>
        
        Returns:
            Dict[int, List[str]]  -> mapping of chunk_number to list of functionalities
        """
        result = {}
        for line in output.strip().splitlines():
            if not line.strip():
                continue
            try:
                chunk_num_str, funcs_str = line.split(":", 1)
                chunk_num = int(chunk_num_str.strip())
                funcs = [f.strip() for f in funcs_str.split(";") if f.strip()]
                # Handle "None"
                if len(funcs) == 1 and funcs[0].lower() == "none":
                    funcs = []
                result[chunk_num] = funcs
            except ValueError:
                # Skip malformed lines gracefully
                continue
        return result
                


if __name__ == "__main__":
    folder_path = "batch_results_extra"  # Example folder path
    jsonl_folder = JsonlFolder(folder_path, recursive=True)
    jsonl_folder.read_folder()