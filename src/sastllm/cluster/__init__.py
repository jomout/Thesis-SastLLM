"""
sastllm.cluster
~~~~~~~~~~~~~~~~~~~
This package contains embedding helpers used before clustering.

The clustering phase itself lives in `sastllm.clustering`.
"""

from .embedder import Embedder

__all__ = ["Embedder"]
