import hashlib
import re
from typing import Dict, List

import tiktoken

from .code_parser import CodeParser
from .comment_stripper import CommentStripper
from .tree_sitter_generator import TreeSitterGenerator

BLOB_RE = re.compile(
    r"['\"]([A-Za-z0-9+/=\s]{200,})['\"]", 
    re.MULTILINE
)

def scrub_long_strings(text: str) -> str:
    def replacer(m):
        s = m.group(1).replace("\n", "")
        h = hashlib.sha256(s.encode()).hexdigest()[:12]
        return f"[ENCODED_BLOB len={len(s)} sha256={h}]"
    return BLOB_RE.sub(replacer, text)


def _last_nonempty_index(lines: List[str]) -> int:
    """Return zero-based index of last non-empty line, or len(lines)-1 if all empty."""
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            return i
    return len(lines) - 1

def _normalize_whitespace(text: str) -> str:
    """Delete consecutive blank lines and trim leading/trailing whitespace."""
    compact = "\n".join(
        line for line in text.splitlines() if line.strip()
    )
    return re.sub(r'\n+', '\n', compact.replace('\r\n', '\n')).strip()


class CodeChunker:
    """
    Chunks source code into semantically meaningful blocks based on AST structure
    and token count constraints.
    """
    def __init__(
        self,
        *, 
        encoding: str = "cl100k_base",
        max_tokens: int = 100,
        remove_comments: bool = False       
    ) -> None:
        """
        Initializes the CodeChunker with a token budget and encoding scheme.

        Args:
            encoding (Optional[str]): Encoding name for tokenization.
            max_tokens (Optional[int]): Maximum token count per chunk.
        """
        self.tree_sitter_gen = TreeSitterGenerator(yaml_config="configs/languages.yaml")
        self.parser = CodeParser(self.tree_sitter_gen)
        self.comment_stripper = CommentStripper(self.tree_sitter_gen)
        self.encoding = encoding
        self.max_tokens = max_tokens
        
        self.remove_comments = remove_comments

    def _count_tokens(self, string: str, encoding_name: str) -> int:
        """
        Counts the number of tokens in a string using the specified encoding.

        Args:
            string (str): The input string.
            encoding_name (str): The encoding name to use.

        Returns:
            int: Number of tokens.
        """
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
        
    def chunk_code(self, code: str, language: str) -> Dict:
        """
        Chunks the input code into segments constrained by token limits
        and guided by syntax-aware breakpoints.

        Args:
            code (str): The full source code string.
            file_extension (str): File extension to infer language.

        Returns:
            Dict[int, str]: A dictionary of code chunks.
        """
        try:
            breakpoints, comments = self.parser.get_lines(code=code, language=language)
        except Exception as e:
            raise RuntimeError(f"Failed to parse code for language '{language}': {e}")

        # We work in zero-based indices for slicing; incoming breakpoints/comments are assumed 1-based.
        # If your parser already returns 0-based, delete the -1/+1 conversions below.
        comment_set_0 = {c - 1 for c in comments}
        lines = code.split("\n")

        # Adjust breakpoints upward to include preceding contiguous comment block
        adjusted = []
        for bp in breakpoints:
            bp0 = bp - 1  # zero-based
            j = bp0 - 1
            top = bp0
            while j >= 0 and j in comment_set_0:
                top = j
                j -= 1
            adjusted.append(top + 1)  # back to 1-based for set uniqueness, then we’ll convert again

        # Unique + sorted, then convert to zero-based for comparisons
        breakpoints_0 = sorted({b - 1 for b in adjusted if b > 0})

        chunks = {}
        chunk_no = 0
        start_idx = 0         # zero-based inclusive
        token_budget = 0
        i = 0

        while i < len(lines):
            # Remove nonsense long strings to avoid bloating token counts
            lines[i] = scrub_long_strings(lines[i])
            line = lines[i]

            new_tokens = self._count_tokens(line, self.encoding)

            if token_budget + new_tokens > self.max_tokens:
                # choose an end exclusive index using the nearest breakpoint <= i
                if i in breakpoints_0:
                    end_excl = i + 1
                else:
                    prev_bps = [x for x in breakpoints_0 if x < i]
                    end_excl = (max(prev_bps) + 1) if prev_bps else start_idx  # end exclusive

                if end_excl == start_idx:
                    # No safe breakpoint yet; force-include this long line
                    token_budget += new_tokens
                    i += 1
                    continue

                # Emit chunk
                chunk_text = "\n".join(lines[start_idx:end_excl]).rstrip("\n")
                
                if self.remove_comments:
                    chunk_text = self.comment_stripper.strip(chunk_text, language=language)

                chunk_text = _normalize_whitespace(chunk_text)

                if chunk_text.strip():
                    chunks[chunk_no] = {
                        "code": chunk_text,
                        "start_line": start_idx + 1,  # 1-based
                        "end_line": end_excl,         # inclusive in 1-based
                    }
                    chunk_no += 1

                # Reset for next chunk
                start_idx = end_excl
                token_budget = 0
            else:
                token_budget += new_tokens
                i += 1

        # Emit trailing chunk if any lines remain
        if start_idx < len(lines):
            # determine a clean inclusive end index (avoid counting a trailing empty split artifact)
            last_idx = _last_nonempty_index(lines)
            end_excl = last_idx + 1
            if end_excl > start_idx:
                chunk_text = "\n".join(lines[start_idx:end_excl]).rstrip("\n")
                
                if self.remove_comments:
                    chunk_text = self.comment_stripper.strip(chunk_text, language=language)
                    
                chunk_text = _normalize_whitespace(chunk_text)

                if chunk_text.strip():
                    chunks[chunk_no] = {
                        "code": chunk_text,
                        "start_line": start_idx + 1,
                        "end_line": end_excl,
                    }

        return chunks