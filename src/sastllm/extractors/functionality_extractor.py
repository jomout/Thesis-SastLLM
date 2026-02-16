import re
from typing import Dict, List, Optional


class FunctionalityExtractor:
    """
    Extracts structured malware behavioral functionality descriptions
    from LLM-generated analysis reports.

    This class assumes the LLM output is formatted in sections labeled as:
        **Chunk N**
        Functionalities: [text describing behaviors]

    It processes the text to isolate functionality descriptions for each chunk,
    and splits them into individual behavior statements.
    """

    def _extract_chunk_sections(self, text: str) -> List[str]:
        """
        Identifies and isolates chunk sections from the full LLM response.

        Args:
            text (str): The full LLM-generated response containing multiple chunk blocks.

        Returns:
            List[str]: A list of chunk sections, each corresponding to a "**Chunk N**" block.
        """
        return re.findall(r"\*\*Chunk \d+\*\*.*?(?=\*\*Chunk \d+\*\*|$)", text, re.DOTALL)

    def _extract_chunk_id(self, section: str) -> Optional[str]:
        """
        Extracts the chunk identifier from a section (e.g., "**Chunk 3**").

        Args:
            section (str): A text block containing chunk metadata and functionality.

        Returns:
            Optional[str]: A string in the format "Chunk N", or None if not found.
        """
        match = re.search(r"\*\*Chunk (\d+)\*\*", section)
        return f"Chunk {match.group(1)}" if match else None

    def _extract_functionality(self, section: str) -> str:
        """
        Extracts the paragraph of functionality text following 'Functionalities:'.

        Args:
            section (str): A single chunk's text block from the LLM response.

        Returns:
            str: The raw paragraph describing functionalities, or an empty string if missing.
        """
        match = re.search(r"Functionalities:\s*(.+)", section, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Naively splits a paragraph of functionality text into individual sentences.

        Args:
            text (str): Raw paragraph text containing one or more actions.

        Returns:
            List[str]: A list of sentences describing separate behaviors.
        """
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in raw_sentences if s]

    def extract_all(self, llm_text: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Processes the full LLM response and extracts structured functionality data
        per chunk, splitting each behavior into a separate sentence.

        Args:
            llm_text (str): Full LLM-generated output containing multiple chunk sections.

        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary mapping chunk IDs to a list of
                                             behavioral functionality sentences. Format:
            {
                "Chunk 0": {
                    "functionalities": [
                        "Loads a malicious DLL.",
                        "Injects code into browser processes.",
                        ...
                    ]
                },
                ...
            }
        """
        results = {}
        for section in self._extract_chunk_sections(llm_text):
            chunk_id = self._extract_chunk_id(section)
            if not chunk_id:
                continue

            paragraph = self._extract_functionality(section)
            functionalities = self._split_into_sentences(paragraph)

            results[chunk_id] = {
                "functionalities": functionalities
            }

        return results
