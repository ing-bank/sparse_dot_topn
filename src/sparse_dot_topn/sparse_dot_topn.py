# Copyright (c) 2023 ING Analytics Wholesale Banking
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from sparse_dot_topn.lib import _sparse_dot_topn_core as _core
from sparse_dot_topn.types import assert_idx_dtype, assert_supported_dtype, ensure_compatible_dtype

if TYPE_CHECKING:
    from numpy.types import DTypeLike, NDArray

__all__ = ["sparse_dot_topn", "awesome_cossim_topn"]


_SUPPORTED_DTYPES = {np.dtype("int32"), np.dtype("int64"), np.dtype("float32"), np.dtype("float64")}


def awesome_cossim_topn(*args, **kwargs):
    """This function has been removed and replaced with `sparse_dot_topn`."""
    msg = "`awesome_cossim_topn` function has been removed and replaced with `sparse_dot_topn`."
    raise NotImplementedError(msg)


def sparse_dot_topn(
    A: csr_matrix | csc_matrix | coo_matrix,
    B: csr_matrix | csc_matrix | coo_matrix,
    top_n: int,
    threshold: float | None = None,
    n_threads: int | None = None,
    idx_dtype: DTypeLike | None = None,
) -> csr_matrix | tuple[csr_matrix, NDArray]:
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
        C: result matrix (returned alone, if return_best_ntop=False)
        nz_counts (optional): the number of elements in A * B that exceeded the threshold for each row of A

    """
    threshold: float = threshold or 0.0
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
        if isinstance(B, (coo_matrix, csr_matrix)):
            B = B.tocsc(False)
    elif A_ncols == B_ncols:
        B = B.transpose() if isinstance(B, csr_matrix) else B.transpose().tocsc(False)
        B_nrows, B_ncols = B.shape
    else:
        msg = (
            "Matrices `A` and `B` have incompatible shapes. `A.shape[1]` must be equal to `B.shape[0]` or `B.shape[1]`."
        )
        raise ValueError(msg)

    assert_supported_dtype(A)
    assert_supported_dtype(B)
    ensure_compatible_dtype(A, B)

    # the max number of result entries
    max_nz = A_nrows * top_n

    # basic check. if A or B are all zeros matrix, return all zero matrix directly
    if A.indices.size == 0 or B.indices.size == 0:
        indptr = np.zeros(A_nrows + 1, dtype=idx_dtype)
        indices = np.zeros(max_nz, dtype=idx_dtype)
        data = np.zeros(max_nz, dtype=A.dtype)
        return csr_matrix((data, indices, indptr), shape=(A_nrows, B_ncols))

    indptr = np.empty(A_nrows + 1, dtype=idx_dtype)

    # filled matrices from here on
    indices = np.empty(max_nz, dtype=idx_dtype)
    data = np.empty(max_nz, dtype=A.dtype)

    best_ntop_arr = np.zeros(A_nrows, dtype=idx_dtype)

    if n_threads > 1:
        warnings.warn("multithreading is currently not supported, reverting to sequential mode.")

    alt_indices, alt_data = _core.sparse_dot_topn(
        A_nrows,
        B_ncols,
        np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        top_n,
        threshold,
        indptr,
        indices,
        data,
        best_ntop_arr,
    )

    if alt_indices is not None:
        indices = alt_indices
        data = alt_data

    return csr_matrix((data, indices, indptr), shape=(A_nrows, B_ncols))
