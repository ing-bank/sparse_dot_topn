# Copyright (c) 2023 ING Analytics Wholesale Banking
import importlib.metadata
import os

# Setting the following environment variable allows multiple OpenMP
# libraries to be loaded. This is also used without issue by Scikit-learn.
# OpenMP error msg:
# /* OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
#  * OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
#  * That is dangerous, since it can degrade performance or cause incorrect results.
#  * The best thing to do is to ensure that only a single OpenMP runtime is linked into the process,
#  * e.g. by avoiding static linking of the OpenMP runtime in any library.
#  * As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE
#  * to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results.
#  * For more information, please see http://openmp.llvm.org/
#  */
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Workaround issue discovered in intel-openmp 2019.5:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

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
