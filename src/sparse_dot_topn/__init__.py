# Copyright (c) 2023 ING Analytics Wholesale Banking
import importlib.metadata

__version__ = importlib.metadata.version("sparse_dot_topn")
from sparse_dot_topn.api import awesome_cossim_topn, sp_matmul, sp_matmul_topn, zip_sp_matmul_topn
from sparse_dot_topn.lib import _sparse_dot_topn_core as _core
from sparse_dot_topn.lib._sparse_dot_topn_core import _has_openmp_support

__all__ = [
    "awesome_cossim_topn",
    "sp_matmul",
    "sp_matmul_topn",
    "zip_sp_matmul_topn",
    "_core",
    "__version__",
    "_has_openmp_support",
]
