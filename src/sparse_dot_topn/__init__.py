# Copyright (c) 2023 ING Analytics Wholesale Banking
import importlib.metadata

__version__ = importlib.metadata.version("sparse_dot_topn")
from sparse_dot_topn.lib import _sparse_dot_topn_core as _core

__all__ = ["_core", "__version__"]
