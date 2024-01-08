# Copyright (c) 2023 ING Analytics Wholesale Banking
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from sparse_dot_topn.lib import _sparse_dot_topn_core as _core
from sparse_dot_topn.types import assert_idx_dtype, assert_supported_dtype, ensure_compatible_dtype

if TYPE_CHECKING:
    from numpy.types import DTypeLike

__all__ = ["sp_matmul_topn", "awesome_cossim_topn"]


_SUPPORTED_DTYPES = {np.dtype("int32"), np.dtype("int64"), np.dtype("float32"), np.dtype("float64")}


def awesome_cossim_topn(
    A, B, ntop, lower_bound=0, use_threads=False, n_jobs=1, return_best_ntop=None, test_nnz_max=None
):
    """This function has been removed and replaced with `sp_matmul_topn`.

    NOTE this function calls `sp_matmul_topn` but the results may not be the same.
    See the migration guide at 'https://github.com/ing-bank/sparse_dot_topn#migration' for details.
    """
    msg = (
        "`awesome_cossim_topn` function has been removed and (partially) replaced with `sp_matmul_topn`."
        " See the migration guide at 'https://github.com/ing-bank/sparse_dot_topn#migration'."
    )
    if return_best_ntop is True or test_nnz_max is not None:
        raise DeprecationWarning(msg)
    msg += " Calling `sp_matmul_topn`, WARNING the results may not be the same."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    n_threads = n_jobs if use_threads is True else None
    return sp_matmul_topn(A=A, B=B, top_n=ntop, threshold=lower_bound, n_threads=n_threads)


def sp_matmul(*args, **kwargs):
    msg = "Sparse Matrix Multiplication is not yet implemented."
    raise NotImplementedError(msg)


def sp_matmul_topn(
    A: csr_matrix | csc_matrix | coo_matrix,
    B: csr_matrix | csc_matrix | coo_matrix,
    top_n: int,
    threshold: int | float | None = None,
    density: float | None = None,
    n_threads: int | None = None,
    idx_dtype: DTypeLike | None = None,
) -> csr_matrix:
    """Compute A * B whilst only storing the `top_n` elements.

    This functions allows large matrices to multiplied with a limited memory footprint.

    Note that

    Args:
        A: LHS of the multiplication, the number of columns of A determines the orientation of B.
            `A` must be have an {32, 64}bit {int, float} dtype that is of the same kind as `B`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        B: RHS of the multiplication, the number of rows of B must match the number of columns of A or the shape of B.T should be match A.
            `B` must be have an {32, 64}bit {int, float} dtype that is of the same kind as `A`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        top_n: the number of results to retain
        threshold: only return values greater than the threshold, by default this 0.0
        density: the expected density of the result considering `top_n`. The expected number of non-zero elements
            in C should <= (`density` * `top_n` * `A.shape[0]`) otherwise the memory has to reallocated.
            This value should only be set if you have a strong expectation as being wrong incurs a realloaction penalty.
        n_threads: number of threads to use, `None` implies sequential processing, -1 will use all but one of the available cores.
        idx_dtype: dtype to use for the indices, defaults to 32bit integers

    Throws:
        TypeError: when A, B are not trivially convertable to a `CSR matrix`

    Returns:
        C: result matrix

    """
    n_threads: int = n_threads or 1
    density: float = density or 1.0
    idx_dtype = assert_idx_dtype(idx_dtype)

    if isinstance(A, csc_matrix) and isinstance(B, csc_matrix) and A.shape[0] == B.shape[1]:
        A = A.transpose()
        B = B.transpose()
    elif isinstance(A, (coo_matrix, csc_matrix)):
        A = A.tocsr(False)
    elif not isinstance(A, csr_matrix):
        msg = f"type of `A` must be one of `csr_matrix`, `csc_matrix` or `csr_matrix`, got `{type(A)}`"
        raise TypeError(msg)

    if not isinstance(B, (csr_matrix, coo_matrix, csc_matrix)):
        msg = f"type of `B` must be one of `csr_matrix`, `csc_matrix` or `csr_matrix`, got `{type(B)}`"
        raise TypeError(msg)

    A_nrows, A_ncols = A.shape
    B_nrows, B_ncols = B.shape

    if A_ncols == B_nrows:
        if isinstance(B, (coo_matrix, csc_matrix)):
            B = B.tocsr(False)
    elif A_ncols == B_ncols:
        B = B.transpose() if isinstance(B, csc_matrix) else B.transpose().tocsr(False)
        B_nrows, B_ncols = B.shape
    else:
        msg = (
            "Matrices `A` and `B` have incompatible shapes. `A.shape[1]` must be equal to `B.shape[0]` or `B.shape[1]`."
        )
        raise ValueError(msg)

    assert_supported_dtype(A)
    assert_supported_dtype(B)
    ensure_compatible_dtype(A, B)

    # guard against top_n larger than number of cols
    top_n = min(top_n, B_ncols)

    # handle threshold
    if np.issubdtype(A.data.dtype, np.integer):
        threshold = int(np.rint(threshold)) if threshold is not None else np.iinfo(A.data.dtype).min
    else:
        threshold = threshold if threshold is not None else np.finfo(A.data.dtype).min

    # basic check. if A or B are all zeros matrix, return all zero matrix directly
    if A.indices.size == 0 or B.indices.size == 0:
        C_indptr = np.zeros(A_nrows + 1, dtype=idx_dtype)
        C_indices = np.zeros(1, dtype=idx_dtype)
        C_data = np.zeros(1, dtype=A.dtype)
        return csr_matrix((C_data, C_indices, C_indptr), shape=(A_nrows, B_ncols))

    kwargs = {
        "top_n": top_n,
        "nrows": A_nrows,
        "ncols": B_ncols,
        "threshold": threshold,
        "density": density,
        "A_data": A.data,
        "A_indptr": A.indptr if idx_dtype is None else A.indptr.astype(idx_dtype),
        "A_indices": A.indices if idx_dtype is None else A.indices.astype(idx_dtype),
        "B_data": B.data,
        "B_indptr": B.indptr if idx_dtype is None else B.indptr.astype(idx_dtype),
        "B_indices": B.indices if idx_dtype is None else B.indices.astype(idx_dtype)
    }

    func = _core.sp_matmul_topn
    if n_threads > 1:
        if _core._has_openmp_support:
            kwargs["n_threads"] = n_threads
            kwargs.pop("density")
            func = _core.sp_matmul_topn_mt
        else:
            msg = "sparse_dot_topn: extension was compiled without parallelisation (OpenMP) support, ignoring ``n_threads``"
            warnings.warn(msg, stacklevel=1)
    return csr_matrix(func(**kwargs), shape=(A_nrows, B_ncols))
