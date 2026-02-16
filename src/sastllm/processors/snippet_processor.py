import json
import os
import random
from itertools import islice
from time import sleep
from typing import Any, Dict, Iterator, List, Tuple

from tqdm import tqdm

from sastllm.analyzers import FunctionalityAnalyzer
from sastllm.configs import get_logger
from sastllm.db import FunctionalityManager, SnippetManager
from sastllm.dtos import CreateFunctionalityDto, GetExtendedSnippetDto, UpdateSnippetDto
from sastllm.parsers import TreeSitterGenerator
from sastllm.prompt import FunctionalityPromptGenerator
from sastllm.utils import Normalizer

logger = get_logger(__name__)

class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass

def retry(max_retries=10, delay=1, backoff=1.5, max_delay=60, jitter=0.1, exceptions=(Exception,), logger=None):
    """
    Retry decorator with capped exponential backoff and optional jitter.

    Args:
        max_retries (int): Maximum number of attempts.
        delay (float): Initial delay between retries (seconds).
        backoff (float): Backoff growth factor (1.5 is gentle, 2.0 is steep).
        max_delay (float): Upper limit for delay.
        jitter (float): Random jitter factor (0.1 = ±10%).
        exceptions (tuple): Exceptions to catch and retry.
        logger (logging.Logger): Optional logger for messages.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        if logger:
                            logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    if logger:
                        logger.warning(f"{func.__name__} failed (attempt {attempt}/{max_retries}): {e}")

                    # Add jitter and cap the delay
                    sleep_time = min(current_delay, max_delay) * random.uniform(1 - jitter, 1 + jitter)
                    sleep(sleep_time)

                    # Gradually increase delay, capped
                    current_delay = min(current_delay * backoff, max_delay)
        return wrapper
    return decorator



class SnippetProcessor:
    """
    Orchestrates the generation and extraction of high-level functionalities from a collection of code snippets.

    This class serves as a pipeline for analyzing code snippets, extracting
    their functionalities using a large language model (LLM), normalizing
    the extracted descriptions, and persisting the results in a database.
    """

    def __init__(
        self,
        *,
        llm,
        batch_size: int = 50,
        sleep_interval: int = 10
    ) -> None:
        """
        Initialize the SnippetProcessor with its dependencies and configuration.

        Args:
            llm: An LLM client or wrapper used to analyze code snippet prompts.
            db: Database manager for fetching and storing functionalities.
            batch_size (int, optional): Number of snippets processed per batch. Defaults to 50.
            sleep_interval (int, optional): Delay between batch processing (in seconds). Defaults to 10.
        """
        logger.debug("Initializing SnippetProcessor.")

        self.llm = llm
        
        self.batch_size = batch_size
        self.sleep_interval = sleep_interval

        self.tree_sitter_gen = TreeSitterGenerator(yaml_config="configs/languages.yaml")

        self.gen = FunctionalityPromptGenerator(tree_sitter_gen=self.tree_sitter_gen)
        self.analyzer = FunctionalityAnalyzer(llm=llm)
        
        self.functionality_db = FunctionalityManager()
        self.snippet_db = SnippetManager()
        
        self.normalizer = Normalizer()
        
        logger.debug("SnippetProcessor initialized.")


    def run_batch(self, code_snippets: List[GetExtendedSnippetDto]) -> None:
        """
        Processes a single batch of code snippets through the full pipeline.

        Steps:
            1. Generate a prompt for the LLM using the given snippets.
            2. Analyze the prompt with the LLM to get raw output.
            3. Extract structured functionality data from the output.
            4. Optionally print verbose debugging information.
            5. Normalize descriptions and store them in the database.

        Args:
            code_snippets (List[GetExtendedSnippetDto]): List of code snippet objects to analyze.
        """
        logger.debug(f"Generating functionalities for {len(code_snippets)} code snippets.")
        
        try:
            # Generate Prompt
            snippets_prompt = self.gen.generate_prompt(code_snippets)
            try:
                # Analyze Prompt
                response = self._analyze_snippet(snippets_prompt)
            except KeyboardInterrupt:
                logger.info("LLM analysis interrupted by user.")
                raise
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
                raise LLMError(f"LLM analysis failed: {e}") from e

            # Extract Data
            extracted_response = self._parse_functionality_output(response)
            
            # Store Functionalities
            self._store_functionalities(extracted_response)
            
        except LLMError as llm_err:
            logger.warning(f"Skipping batch due to LLM error: {llm_err}")
            raise llm_err
        except Exception as e:
            logger.warning(f"Failed to process batch: {e}")
            raise RuntimeError(f"Failed to process batch: {e}") from e

        logger.debug("Generated functionalities for %d snippets.", len(code_snippets))

    
    def run(self) -> None:
        """
        Runs the snippet processing pipeline on all available code snippets in batches.

        Fetches code snippets from the database, splits them into batches,
        and processes each batch sequentially. Waits for `sleep_interval`
        seconds between batches.

        Args:
            verbose (bool, optional): If True, enables verbose mode for each batch.
        """
        # Fetch Snippets
        snippet_batches, total_batches = self._fetch_snippets_into_batches()

        logger.info(f"Starting snippet processing in batches of {self.batch_size} - Using llm {self.analyzer.llm._llm_type}")

        for i, batch in tqdm(enumerate(snippet_batches), desc="Analyzing snippets", total=total_batches):
            start = i * self.batch_size + 1
            end = (i + 1) * self.batch_size + 1
            try:
                logger.debug(f"Processing batch [{start}, {end - 1}].")

                if not batch:
                    logger.debug(f"No snippets to process in batch {i}/{total_batches} [{start}, {end - 1}]. Skipping.")
                    continue
                
                # Process Batch
                self.run_batch(batch)
                
                logger.debug(f"Sleeping for {self.sleep_interval} seconds.")
                sleep(self.sleep_interval)
            except KeyboardInterrupt:
                logger.info(f"Processing interrupted by user. Stopped at batch [{start}, {end - 1}].")
                break
            except LLMError as llm_err:
                logger.warning(f"Failed batch {i}/{total_batches}, [{start}, {end - 1}] due to LLM error: {llm_err}")
                break
            except Exception as e:
                logger.warning(f"Failed to process batch {i}/{total_batches}, [{start}, {end - 1}]: {e}")
                break
        
        logger.info("Snippet processing completed successfully.")


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

    
    def _fetch_snippets_into_batches(self) -> Tuple[Iterator[List[GetExtendedSnippetDto]], int]:
        """
        Fetches code snippets from the database and organizes them into batches.
        Returns:
            Tuple[Iterator[List[GetExtendedSnippetDto]], int]: An iterator over batches of code snippets and the total number of batches.
        """
        # Fetch all unprocessed code snippets with file metadata
        snippets = self.snippet_db.get_snippets_with_file_meta(batch_size=self.batch_size)

        total_snippets = self.snippet_db.get_num_snippets()
        total_batches = (total_snippets + self.batch_size - 1) // self.batch_size

        return self._batch_objects(snippets, self.batch_size), total_batches
        

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
                snippet = UpdateSnippetDto(snippet_id=snippet_id, processed=True)
                self.snippet_db.update_snippet(snippet=snippet)

                # Write to file for debugging - caching
                self._cache(dir=f"cache/functionalities-{self.analyzer.llm._llm_type}", snippet_id=snippet_id, payload=payload)

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


    @retry(max_retries=1000, delay=1, logger=logger)
    def _analyze_snippet(self, snippet_prompt: str) -> str:
        """
        Analyzes a multi-snippet prompt and returns the LLM's response.

        Args:
            snippet_prompt (str): The code snippets to analyze.

        Returns:
            str: The LLM's response.
        """
        return self.analyzer.analyze(snippet_prompt)