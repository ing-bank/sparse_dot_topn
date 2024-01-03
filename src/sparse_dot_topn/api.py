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


def awesome_cossim_topn(*args, **kwargs):
    """This function has been removed and replaced with `sp_matmul_topn`."""
    msg = "`awesome_cossim_topn` function has been removed and replaced with `sp_matmul_topn`."
    raise NotImplementedError(msg)


def sp_matmul(*args, **kwargs):
    msg = "Sparse Matrix Multiplication is not yet implemented."
    raise NotImplementedError(msg)


def sp_matmul_topn(
    A: csr_matrix | csc_matrix | coo_matrix,
    B: csr_matrix | csc_matrix | coo_matrix,
    top_n: int,
    threshold: int | float | None = None,
    n_threads: int | None = None,
    idx_dtype: DTypeLike | None = None,
) -> csr_matrix:
    """Compute A * B whilst only storing the `top_n` elements.

    This functions allows large matrices to multiplied with a limited memory footprint.

    Args:
        A: LHS of the multiplication, the number of columns of A determines the orientation of B.
            `A` must be have an {32, 64}bit {int, float} dtype that is of the same kind as `B`.
        B: RHS of the multiplication, the number of rows of B must match the number of columns of A or the shape of B.T should be match A.
            `B` must be have an {32, 64}bit {int, float} dtype that is of the same kind as `A`.
        top_n: the number of results to retain
        threshold: only return values greater than the threshold, by default this 0.0
        n_threads: number of threads to use, `None` implies sequential processing, -1 will use all but one of the available cores.
        idx_dtype: dtype to use for the indices, defaults to 32bit integers

    Throws:
        TypeError: when A, B are not trivially convertable to a `CSR matrix`

    Returns:
        C: result matrix

    """
    n_threads: int = n_threads or 1
    idx_dtype = assert_idx_dtype(idx_dtype)

    if isinstance(A, (coo_matrix, csc_matrix)):
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

    # handle threshold
    if np.issubdtype(A.data.dtype, np.integer):
        threshold = int(np.rint(threshold)) if threshold is not None else np.iinfo(A.data.dtype).min
    else:
        threshold = threshold if threshold is not None else np.finfo(A.data.dtype).min

    # the max number of result entries
    max_nz = A_nrows * top_n

    # basic check. if A or B are all zeros matrix, return all zero matrix directly
    if A.indices.size == 0 or B.indices.size == 0:
        C_indptr = np.zeros(A_nrows + 1, dtype=idx_dtype)
        C_indices = np.zeros(max_nz, dtype=idx_dtype)
        C_data = np.zeros(max_nz, dtype=A.dtype)
        return csr_matrix((C_data, C_indices, C_indptr), shape=(A_nrows, B_ncols))

    C_indptr = np.zeros(A_nrows + 1, dtype=idx_dtype)
    C_indices = np.zeros(max_nz, dtype=idx_dtype)
    C_data = np.zeros(max_nz, dtype=A.dtype)

    if n_threads > 1:
        warnings.warn("multithreading is currently not supported, reverting to sequential mode.")

    _core.sp_matmul_topn(
        top_n,
        A_nrows,
        B_ncols,
        threshold,
        A.data,
        A.indptr if idx_dtype is None else A.indptr.astype(idx_dtype),
        A.indices if idx_dtype is None else A.indices.astype(idx_dtype),
        B.data,
        B.indptr if idx_dtype is None else B.indptr.astype(idx_dtype),
        B.indices if idx_dtype is None else B.indices.astype(idx_dtype),
        C_data,
        C_indptr,
        C_indices,
    )
    return csr_matrix((C_data, C_indices, C_indptr), shape=(A_nrows, B_ncols))
