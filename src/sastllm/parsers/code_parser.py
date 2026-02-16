import os
from typing import Dict, List, Tuple

import yaml
from tree_sitter import Node, Parser


class CodeParser:
    """
    A class for parsing source code using Tree-sitter and extracting important 
    semantic elements (e.g., classes, functions, comments) depending on the language.
    """
    
    IMPORTANT_NODE_TYPES: Dict[str, Dict[str, str]] = {}
    COMMENT_NODES: Dict[str, Dict[str, str]] = {}

    def __init__(self, tree_sitter_gen) -> None:
        """
        Initializes the CodeParser.
        """
        self.tree_sitter_gen = tree_sitter_gen
        # Load node type mappings from YAML
        self.IMPORTANT_NODE_TYPES = self._load_yaml_mapping("configs/important_nodes.yaml")
        self.COMMENT_NODES = self._load_yaml_mapping("configs/comment_nodes.yaml")

    @staticmethod
    def _load_yaml_mapping(path: str) -> Dict[str, Dict[str, str]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Normalize keys just in case
        normal: Dict[str, Dict[str, str]] = {}
        for lang, mapping in data.items():
            normal[str(lang).lower()] = {str(k): str(v) for k, v in (mapping or {}).items()}
        return normal
    

    def _get_parser(self, language: str) -> Parser:
        """
        Configures the Tree-sitter parser for the specified file extension.

        Args:
            language (str): The programming language (e.g., 'python', 'javascript').

        Raises:
            ValueError: If the language is unsupported.
        """
        return self.tree_sitter_gen.get_parser(language=language)


    def _get_nodes_types(self, language: str, node_types: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """
        Gets the relevant node types for the given language.

        Args:
            language (str): The programming language (e.g., 'python', 'javascript').
            node_types (Dict[str, Dict[str, str]]): A mapping of node types per language.

        Returns:
            Dict[str, str]: The node types relevant for the language.

        Raises:
            ValueError: If the language is unsupported.
        """
        return node_types[language]


    def _extract_nodes(
        self,
        node: Node, 
        language: str,
        node_types: Dict[str, Dict[str, str]],
    ) -> List[Tuple[Node, str]]:
        """
        Recursively extracts relevant AST nodes from the Tree-sitter node tree.

        Args:
            node (Node): The root or current AST node.
            language (str): The programming language (e.g., 'python', 'javascript').
            node_types (Dict[str, Dict[str, str]]): Node types to extract.

        Returns:
            List[Tuple[Node, str]]: A list of (node, label) tuples.
        """
        # Fetch important node types for given language
        types = self._get_nodes_types(
            language=language,
            node_types=node_types
        )
        
        # Collect node types
        nodes = []
        if node.type in types:
            if language in ('c', 'cpp') and node.type == 'struct_specifier':
                has_body = any(child.type == 'field_declaration_list' for child in node.children)
                if has_body:
                    nodes.append((node, types[node.type]))
            else:
                nodes.append((node, types[node.type]))

        for child in node.children:
            nodes.extend(
                self._extract_nodes(
                    node=child,
                    language=language,
                    node_types=node_types
                )
            )

        return nodes

    
    def get_lines(
        self, 
        code: str, 
        language: str
    ) -> Tuple[List[int], List[int]]:
        """
        Extracts line numbers of important and comment nodes from source code.

        Args:
            code (str): The source code string.
            file_extension (str): The file extension (e.g., 'py', 'js').

        Returns:
            Tuple[List[int], List[int]]: A tuple of line numbers:
                - Important code nodes (functions, classes, etc.)
                - Comments and decorators
        """
        # Configure Parser
        parser = self._get_parser(
            language=language
        )
        
        # Parse code
        tree = parser.parse(code.encode('utf-8'))

        root_node = tree.root_node
        
        important_nodes = self._extract_nodes(
            node=root_node,
            language=language,
            node_types=self.IMPORTANT_NODE_TYPES
        )
        
        comment_nodes = self._extract_nodes(
            node=root_node,
            language=language,
            node_types=self.COMMENT_NODES
        )
        
        lines = []
        
        for nodes in [important_nodes, comment_nodes]:
            
            important_lines = {}

            for node, interest in nodes:
                start_line = node.start_point[0] 
                if interest not in important_lines:
                    important_lines[interest] = []

                if start_line not in important_lines[interest]:
                    important_lines[interest].append(start_line)

            lines_of_interest = []
            for _, line_numbers in important_lines.items():
                lines_of_interest.extend(line_numbers)
        
            lines.append(sorted(lines_of_interest))


        return lines[0], lines[1]