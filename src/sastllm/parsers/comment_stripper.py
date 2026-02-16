from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from tree_sitter import Node, Parser


@dataclass
class StripOptions:
    preserve_shebang: bool = True
    # For languages that treat top-of-file directives as comments you may want to keep:
    preserve_license_header_lines: int = 0  # e.g., keep first N lines


class CommentStripper:
    """
    Remove comments from source code using Tree-sitter parse trees.

    - Works across many languages using `tree_sitter_languages` prebuilt grammars.
    - Correctly ignores comment-like tokens inside strings, regexes, etc.
    - Preserves an optional shebang and/or the first N lines (e.g., license banner).

    Usage:
        stripper = CommentStripper()
        cleaned = stripper.strip(code_string, language="python")
        # or let it infer from filename:
        cleaned = stripper.strip(code_string, filename="script.py")
    """

    def __init__(self, tree_sitter_gen) -> None:
        self._parser_cache: Dict[str, Parser] = {}
        self._ts_generator = tree_sitter_gen

    def strip(
        self,
        code: str,
        *,
        language: Optional[str] = None,
        filename: Optional[str] = None,
        options: Optional[StripOptions] = None,
    ) -> str:
        """
        Strip comments from `code`. Either `language` or `filename` must be provided.

        :param code: Source code as a Python str.
        :param language: Tree-sitter language name (e.g. 'python', 'javascript', 'c').
        :param filename: Used only to infer language from extension if `language` is not given.
        :param options: StripOptions controlling preservation of shebang/license header.
        """
        if not language:
            raise ValueError(f"Unable to infer language from filename '{filename}'.")

        options = options or StripOptions()

        parser = self._get_parser(language)

        tree = parser.parse(code.encode("utf-8"))
        if not tree:
            raise ValueError("AST Tree cannot be generated.")
        
        root = tree.root_node

        # Compute deletion ranges (byte offsets) for comment nodes
        delete_spans: List[Tuple[int, int]] = []
        for node in self._walk_comment_nodes(root):
            start_byte = node.start_byte
            end_byte = node.end_byte
            delete_spans.append((start_byte, end_byte))

        # Optional: preserve shebang (#!...) at very start of file
        if options.preserve_shebang and code.startswith("#!"):
            first_newline = code.find("\n")
            if first_newline != -1:
                # Protect shebang span from deletion
                shebang_bytes = len(code[: first_newline + 1].encode("utf-8"))
                delete_spans = self._exclude_range(delete_spans, 0, shebang_bytes)

        # Optional: preserve first N lines (e.g., license header comments)
        if options.preserve_license_header_lines > 0:
            keep_upto_line = options.preserve_license_header_lines
            cutoff_index = self._byte_index_after_n_lines(code, keep_upto_line)
            if cutoff_index is not None:
                delete_spans = self._exclude_range(delete_spans, 0, cutoff_index)

        # Merge overlapping spans to make deletion linear
        delete_spans = self._merge_spans(sorted(delete_spans))

        # Perform deletion in one pass from the end
        mutable = bytearray(code.encode("utf-8"))
        for start, end in reversed(delete_spans):
            del mutable[start:end]

        return mutable.decode("utf-8")

    # ------------------------ internals ------------------------

    def _get_parser(self, language: str) -> Parser:
        lang_key = language.strip().lower()
        if lang_key in self._parser_cache:
            return self._parser_cache[lang_key]
        try:
            parser = self._ts_generator.get_parser(language=lang_key)
        except Exception as e:
            raise ValueError(
                f"Unsupported or unknown language '{language}'. Ensure it exists in configs/languages.yaml. Original error: {e}"
            )
        self._parser_cache[lang_key] = parser
        return parser

    def _walk_comment_nodes(self, node: Node) -> Iterable[Node]:
        """
        Yield nodes that are comments. Many grammars standardize on node.type containing 'comment'.
        This captures 'comment', 'line_comment', 'block_comment', 'documentation_comment', etc.
        """
        stack = [node]
        while stack:
            cur = stack.pop()
            # Standard comments
            if "comment" in cur.type:
                yield cur
                continue

            # Python-specific docstrings: top-level or function/class body strings
            if cur.type == "expression_statement" and cur.child_count == 1:
                child = cur.children[0]
                if child.type == "string":
                    yield child
                    continue

            for i in range(cur.child_count - 1, -1, -1):
                stack.append(cur.children[i])


    @staticmethod
    def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        merged: List[Tuple[int, int]] = []
        for s, e in spans:
            if not merged or s > merged[-1][1]:
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        return merged

    @staticmethod
    def _exclude_range(
        spans: List[Tuple[int, int]],
        keep_start: int,
        keep_end: int,
    ) -> List[Tuple[int, int]]:
        """
        Remove any deletions that overlap [keep_start, keep_end).
        If a deletion partially overlaps, trim it.
        """
        out: List[Tuple[int, int]] = []
        for s, e in spans:
            if e <= keep_start or s >= keep_end:
                out.append((s, e))
            else:
                # Overlap: keep parts outside the preserved range
                if s < keep_start:
                    out.append((s, keep_start))
                if e > keep_end:
                    out.append((keep_end, e))
        return out

    @staticmethod
    def _byte_index_after_n_lines(code: str, n_lines: int) -> Optional[int]:
        if n_lines <= 0:
            return None
        idx = 0
        count = 0
        while count < n_lines and idx != -1:
            idx = code.find("\n", idx)
            if idx == -1:
                break
            idx += 1
            count += 1
        return len(code[:idx].encode("utf-8")) if count == n_lines and idx != -1 else None
