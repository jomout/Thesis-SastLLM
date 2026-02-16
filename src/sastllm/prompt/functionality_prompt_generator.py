import re
from typing import Dict, List

from tree_sitter import Node, Tree

from sastllm.configs import get_logger
from sastllm.dtos.get_dtos import GetExtendedSnippetDto
from sastllm.parsers import TreeSitterGenerator

logger = get_logger(__name__)

MAX_LINES = 400
class FunctionalityPromptGenerator:
    """
    Generates enriched, context-aware code snippets from raw source files
    to be used for LLM-based functionality analysis.

    This includes extracting:
    - Function names
    - Function calls
    - Control flow structures
    - String literals
    - AST elements from parsed source code

    Attributes:
        CONTROL_NODES (set): Types of control structures to identify in the AST.
        STRING_NODE_TYPES (set): Types of nodes considered string literals.
        CALL_EXPR_TYPES (set): Types of AST nodes representing function calls.
    """

    CONTROL_NODES = {'if_statement', 'for_statement', 'while_statement', 'switch_statement', 'match_statement'}
    STRING_NODE_TYPES = {'string', 'string_literal'}
    CALL_EXPR_TYPES = {'call_expression', 'call'}

    def __init__(self, tree_sitter_gen: TreeSitterGenerator):
        self._ts_generator = tree_sitter_gen

    def extract_function_name(self, code: str) -> str:
        """
        Extracts the first function name from a block of source code.

        Args:
            code (str): Source code block as a string.

        Returns:
            str: The name of the function, or "unknown" if not found.
        """
        match = re.search(r'\b([a-zA-Z_]\w*)\s*\([^;{]*\)\s*\{', code)
        return match.group(1) if match else "unknown"

    def extract_ast_elements(self, tree: Tree, code: str) -> Dict[str, List[str]]:
        """
        Traverses the AST to extract key elements like function calls,
        control structures, and string literals.

        Args:
            tree (Tree): The Tree-sitter parsed AST.
            code (str): Original source code corresponding to the tree.

        Returns:
            Dict[str, List[str]]: Dictionary with keys:
                - 'function_calls': List of function call identifiers
                - 'control_structures': List of control structure types
                - 'string_literals': List of literal string values
        """
        root_node = tree.root_node
        calls, control, strings = set(), set(), set()

        def walk(node: Node):
            if node.type in self.CALL_EXPR_TYPES:
                callee = node.child_by_field_name('function') or node.child(0)
                if callee:
                    calls.add(code[callee.start_byte:callee.end_byte])
            elif node.type in self.CONTROL_NODES:
                control.add(node.type)
            elif node.type in self.STRING_NODE_TYPES:
                strings.add(code[node.start_byte:node.end_byte])
            for child in node.children:
                walk(child)

        walk(root_node)
        return {
            "function_calls": sorted(calls),
            "control_structures": sorted(control),
            "string_literals": sorted(strings),
        }

    @staticmethod
    def _format_snippet(
        snippet_id: int,
        file_path: str,
        func_name: str,
        language: str,
        ast_info: Dict[str, List[str]],
        code: str,
    ) -> str:
        """
        Formats extracted AST information and code into a clean, structured snippet.

        Returns:
            str: Formatted snippet string for LLM processing.
        """
        def format_list(label: str, items: List[str]) -> str:
            if not items:
                return f"  - {label}: None"
            return f"  - {label}: [{', '.join(items)}]"
        
        def format_header(index: int) -> str:
            return f"Snippet ID: {index}"
        
        # Remove any non-ASCII characters from the code
        code = re.sub(r'[^\x00-\x7F]+', '', code)

        # split code into lines for potential truncation
        code_lines = code.splitlines()
        total_lines = len(code_lines)

        if total_lines > MAX_LINES:
            code = "\n".join(code_lines[:MAX_LINES]) + f"\n... [truncated, total lines: {total_lines}]"



        formatted = (
            f"{format_header(snippet_id)}\n"
            f"File: {file_path}\n"
            f"Function: {func_name}\n"
            f"Language: {language}\n"
            f"AST Elements:\n"
            f"{format_list('Function Calls', ast_info.get('function_calls', []))}\n"
            f"{format_list('Control Structures', ast_info.get('control_structures', []))}\n"
            # f"{format_list('String Literals', ast_info.get('string_literals', []))}\n"
            f"\n"
            f"--- Begin Code ---\n"
            f"{code.strip()}\n"
            f"--- End Code ---"
        )

        return formatted


    def generate_prompt(self, code_snippets: List[GetExtendedSnippetDto]) -> str:
        """
        Processes a list of GetExtendedSnippetDto objects and returns formatted,
        annotated code snippets for downstream LLM analysis.

        Args:
            code_snippets (List[GetExtendedSnippetDto]): List of code snippet objects with full metadata.

        Returns:
            str: A formatted prompt with code snippets with context and AST insights.
        """
        logger.debug(f"Generating prompt for {len(code_snippets)} code snippets.")
        
        snippets = []

        for c in code_snippets:
            snippet_id = c.snippet_id
            code = c.code
            language = c.language
            file_name = c.filename
            file_path = c.filepath

            if not language:
                print(f"Skipping unsupported language: {language} (file: {file_name})")
                continue
            try:
                parser = self._ts_generator.get_parser(language=language)
            except ValueError:
                print(f"Skipping unsupported language: {language} (from language {language})")
                continue

            tree = parser.parse(code.encode('utf-8'))

            ast_info = self.extract_ast_elements(tree, code)
            func_name = self.extract_function_name(code)
            snippet = self._format_snippet(snippet_id, file_path, func_name, language, ast_info, code)

            snippets.append(snippet)
            
        logger.debug("Generated prompt for %d code snippets.", len(code_snippets))

        return "\n\n".join(snippets)
