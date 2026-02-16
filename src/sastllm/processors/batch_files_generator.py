import json
import math
import os
from itertools import islice
from pathlib import Path
from string import Template
from typing import Any, Iterator, List, Tuple

from tqdm import tqdm

from sastllm.configs import get_logger
from sastllm.db import SnippetManager
from sastllm.dtos import GetExtendedSnippetDto
from sastllm.parsers import TreeSitterGenerator
from sastllm.prompt import FunctionalityPromptGenerator

logger = get_logger(__name__)


FUNCTIONALITY_PROMPT = """You are a code functionality analysis expert. Analyze the given code snippets and summarize what each chunk functionally does.

Rules:
- Summarize the functionality into clear, semantic-leveled, goal-oriented actions (e.g., "validate input", "decrypt data", "write file").
- Focus on the intended behavior of the code, not the specific instructions.
- Do not mention registers, variables, opcodes, or syntax-level details.
- Use imperative sentences.
- Each action must be one short sentence (max 20 words).
- Separate functionalities with semicolons.
- Always output one line per snippet.
- If no meaningful functionality is found, write "None" after the snippet id.
- Do not add commentary or explanations.

Output format:
<snippet_id>: <functionality 1>; <functionality 2>; <functionality 3>

Examples:
1: Load configuration file; Decrypt payload; Execute decrypted code
2: Generate random token; Connect to remote server; Send authentication request
3: None

Now analyze the following code snippets and output only in this format:

${code_snippets}
"""


class BatchFilesGenerator:
    def __init__(
        self,
        *,
        model: str = "gpt-5-mini",
        snippet_batch_size: int = 20,
        api_batch_size: int = 500,
    ) -> None:
        logger.debug("Initializing BatchSnippetProcessor.")

        self.api_batch_size = api_batch_size
        self.snippet_batch_size = snippet_batch_size

        self.tree_sitter_gen = TreeSitterGenerator(yaml_config="configs/languages.yaml")

        self.gen = FunctionalityPromptGenerator(tree_sitter_gen=self.tree_sitter_gen)

        self.model = model

        self.snippet_db = SnippetManager()

        logger.debug("SnippetProcessor initialized.")

    def create_snippets_batch(self, code_snippets: List[GetExtendedSnippetDto]) -> str:
        logger.debug(f"Creating snippet batch for {len(code_snippets)} code snippets.")

        # Generate Prompt
        snippets_prompt = Template(FUNCTIONALITY_PROMPT).substitute(
            code_snippets=self.gen.generate_prompt(code_snippets)
        )

        logger.debug(f"Generated snippet batch for {len(code_snippets)} code snippets.")

        return snippets_prompt

    def create_api_batches(self, output_dir: str | Path) -> None:
        """
        Create multiple JSONL batch files, each containing up to `api_batch_size`
        prompts generated from `snippet_batch_size` code snippets per prompt.
        """
        output_directory = Path(output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

        snippet_batches_iter, total_batches = self._fetch_snippets_into_batches()
        total_api_batches = math.ceil(total_batches / self.api_batch_size)

        print(f"Total snippet batches: {total_batches}")
        print(
            f"Creating {total_api_batches} API batch files "
            f"({self.api_batch_size} prompts per .jsonl file)."
        )
        current_api_batch_index = 1
        current_snippet_prompts = []
        prompt_count = 0

        for i, snippet_batch in enumerate(tqdm(snippet_batches_iter, total=total_batches), start=1):
            # Create prompt for this snippet batch
            snippet_prompt = self.create_snippets_batch(snippet_batch)
            current_snippet_prompts.append(snippet_prompt)
            prompt_count += 1

            # Once we reach api_batch_size, write a new jsonl file
            if prompt_count >= self.api_batch_size:
                file_name = os.path.join(output_dir, f"api_batch_{current_api_batch_index}.jsonl")
                self.build_api_batch_jsonl(
                    current_snippet_prompts, output_path=file_name, model=self.model
                )
                with open(
                    f"snippet_prompts_debug_{current_api_batch_index}.txt", "a", encoding="utf-8"
                ) as debug_f:
                    for sp in current_snippet_prompts:
                        debug_f.write(sp + "\n\n---\n\n")
                print(f"Wrote {file_name} with {prompt_count} prompts.")

                # Reset for next file
                current_api_batch_index += 1
                current_snippet_prompts = []
                prompt_count = 0

        # Handle remaining snippets (if not a multiple of api_batch_size)
        if current_snippet_prompts:
            file_name = output_directory / f"api_batch_{current_api_batch_index}.jsonl"
            self.build_api_batch_jsonl(
                current_snippet_prompts, output_path=file_name, model=self.model
            )
            print(f"Wrote final {file_name} with {prompt_count} prompts.")

    def _fetch_snippets_into_batches(self) -> Tuple[Iterator[List[GetExtendedSnippetDto]], int]:
        """
        Fetches code snippets from the database and organizes them into batches.
        Returns:
            Tuple[Iterator[List[GetExtendedSnippetDto]], int]: An iterator over batches of code snippets and the total number of batches.
        """
        # Fetch all unprocessed code snippets with file metadata
        snippets = self.snippet_db.get_snippets_with_file_meta(batch_size=self.snippet_batch_size)

        total_snippets = self.snippet_db.get_num_snippets()
        total_batches = math.ceil(total_snippets / self.snippet_batch_size)

        return self._batch_objects(snippets, self.snippet_batch_size), total_batches

    @staticmethod
    def _batch_objects(iterable: Iterator[Any], batch_size: int):
        """
        Generator that yields successive chunks of a given size from an iterable.

        Args:
            iterable (Iterator[Any]): Input iterable.
            batch_size (int): Number of elements per batch.

        Yields:
            List[Any]: A batch of items.
        """
        iterator = iter(iterable)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    @staticmethod
    def build_api_batch_jsonl(
        snippets_prompts: List[str],
        output_path: str | Path,
        model: str = "gpt-5-mini",
    ):
        """
        Build a JSONL file for the OpenAI Batch API using /v1/responses
        with plain text output for code functionality summarization.
        """

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            for i, snippets_prompt in enumerate(snippets_prompts, start=1):
                record = {
                    "custom_id": f"snippets_batch_{i}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model,
                        "input": snippets_prompt,
                        "prompt_cache_key": "functionality_analysis_v1",
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return str(output_file)
