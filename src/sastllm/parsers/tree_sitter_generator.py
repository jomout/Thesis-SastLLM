import logging
import os
import subprocess
from ctypes import c_void_p, cdll
from platform import system
from tempfile import TemporaryDirectory
from typing import List

import yaml
from tree_sitter import Language, Parser

from sastllm.configs import get_logger

logger = get_logger(__name__)


class TreeSitterGenerator:
    def __init__(self, yaml_config: str, build_dir: str = "build", lib_name: str = "my-languages.so"):
        """
        :param yaml_config: Path to YAML config listing languages + repos
        :param build_dir: Directory for grammars + compiled .so
        :param lib_name: Name of the shared object with all grammars
        """
        self.yaml_config = yaml_config
        self.build_dir = build_dir
        self.lib_path = os.path.join(build_dir, lib_name)
        self.parsers = {}
        os.makedirs(build_dir, exist_ok=True)

        self._load_config()
        self._build_shared_library()
        self._init_parsers()

    def _load_config(self):
        with open(self.yaml_config, "r") as f:
            self.config = yaml.safe_load(f)

    def _build_shared_library(self):
        grammar_paths = []
        for lang in self.config["languages"]:
            name = lang["name"]
            repo = lang["repo"]
            lang_dir = os.path.join(self.build_dir, f"tree_sitter_{name}")

            if not os.path.exists(lang_dir):
                print(f"Cloning grammar for {name}...")
                subprocess.run(["git", "clone", repo, lang_dir], check=True)

            grammar_dir = lang_dir
            # Some repos host multiple grammars, select subdir when provided
            subdir = lang.get("subdir")
            if subdir:
                grammar_dir = os.path.join(lang_dir, subdir)

            grammar_paths.append(grammar_dir)

        logger.debug("Building shared library with:", grammar_paths)
        self._build_library(self.lib_path, grammar_paths)
        
    
    @staticmethod
    def _deprecate(old: str, new: str):
        logging.warning(f"{old} is deprecated. Use {new} instead.")
        

    def _build_library(self, output_path: str, repo_paths: List[str]) -> bool:
        """
        Build a dynamic library at the given path, based on the parser
        repositories at the given paths.

        Returns `True` if the dynamic library was compiled and `False` if
        the library already existed and was modified more recently than
        any of the source files.
        """
        output_mtime = os.path.getmtime(output_path) if os.path.exists(output_path) else 0

        if not repo_paths:
            raise ValueError("Must provide at least one language folder")

        cpp = False
        source_paths = []
        for repo_path in repo_paths:
            src_path = os.path.join(repo_path, "src")
            source_paths.append(os.path.join(src_path, "parser.c"))
            if os.path.exists(os.path.join(src_path, "scanner.cc")):
                cpp = True
                source_paths.append(os.path.join(src_path, "scanner.cc"))
            elif os.path.exists(os.path.join(src_path, "scanner.c")):
                source_paths.append(os.path.join(src_path, "scanner.c"))
        source_mtimes = [os.path.getmtime(__file__)] + [os.path.getmtime(path_) for path_ in source_paths]

        if max(source_mtimes) <= output_mtime:
            return False

        # local import saves import time in the common case that nothing is compiled
        try:
            from distutils.ccompiler import new_compiler
            from distutils.unixccompiler import UnixCCompiler

        except ImportError as err:
            raise RuntimeError(
                "Failed to import distutils. You may need to install setuptools."
            ) from err

        compiler = new_compiler()
        if isinstance(compiler, UnixCCompiler):
            compiler.set_executables(compiler_cxx="c++")

        with TemporaryDirectory(suffix="tree_sitter_language") as out_dir:
            object_paths = []
            for source_path in source_paths:
                if system() == "Windows":
                    flags = None
                else:
                    flags = ["-fPIC"]
                    if source_path.endswith(".c"):
                        flags.append("-std=c11")
                object_paths.append(
                    compiler.compile(
                        [source_path],
                        output_dir=out_dir,
                        include_dirs=[os.path.dirname(source_path)],
                        extra_preargs=flags,
                    )[0]
                )
            compiler.link_shared_object(
                object_paths,
                output_path,
                target_lang="c++" if cpp else "c",
            )
        return True
    

    def load_language(self, lib_path: str, lang: str) -> Language:
        lib = cdll.LoadLibrary(lib_path)
        language_function = getattr(lib, f"tree_sitter_{lang}")
        language_function.restype = c_void_p
        ptr = language_function()
        return Language(ptr)
        

    def _init_parsers(self):
        for lang in self.config["languages"]:
            name = lang["name"]
            lang_obj = self.load_language(self.lib_path, name)
            parser = Parser(lang_obj)
            self.parsers[name] = parser

    def get_parser(self, language: str) -> Parser:
        if language not in self.parsers:
            raise ValueError(f"Language '{language}' not found in config.")
        return self.parsers[language]