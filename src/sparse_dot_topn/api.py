# Copyright (c) 2023 ING Analytics Wholesale Banking
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import psutil
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from sparse_dot_topn.lib import _sparse_dot_topn_core as _core
from sparse_dot_topn.types import assert_idx_dtype, assert_supported_dtype, ensure_compatible_dtype

if TYPE_CHECKING:
    from numpy.types import DTypeLike

__all__ = ["sp_matmul", "sp_matmul_topn", "awesome_cossim_topn"]


_N_CORES = psutil.cpu_count(logical=False) - 1

_SUPPORTED_DTYPES = {np.dtype("int32"), np.dtype("int64"), np.dtype("float32"), np.dtype("float64")}


def awesome_cossim_topn(
    A, B, ntop, lower_bound=0, use_threads=False, n_jobs=1, return_best_ntop=None, test_nnz_max=None
):
    """This function will be removed and replaced with `sp_matmul_topn`.

    NOTE this function calls `sp_matmul_topn` but the results may not be the same.
    See the migration guide at 'https://github.com/ing-bank/sparse_dot_topn#migration' for details.

    This function will return a matrix C in CSR format, where
    C = [sorted top n results > lower_bound for each row of A * B].
    If return_best_ntop=True then best_ntop
    (the true maximum number of elements > lower_bound per row of A * B)
    will also be returned in a tuple together with C as (C, best_ntop).

    Args:
        A: LHS of the multiplication, the number of columns of A determines the orientation of B.
            `A` must be have an {32, 64}bit {int, float} dtype that is of the same kind as `B`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        B: RHS of the multiplication, the number of rows of B must match the number of columns of A or the shape of B.T should be match A.
            `B` must be have an {32, 64}bit {int, float} dtype that is of the same kind as `A`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        ntop: top n results
        lower_bound: a threshold that the element of A*B must be greater than
        use_threads: use multi-thread or not
        n_jobs: number of thread, must be >= 1
        return_best_ntop: (default: False) if True, will return best_ntop together
                          with C as a tuple: (C, best_ntop)
        test_nnz_max: deprecated argument, cannot be used

    Returns:
        C: result matrix (returned alone, if return_best_ntop=False)
        best_ntop: The true maximum number of elements > lower_bound per row of
                   A * B returned together with C as a tuple: (C, best_ntop). It is
                   returned only if return_best_ntop=True.

    N.B. if A and B are not in CSR format, they will be converted to CSR
    """
    msg = (
        "`awesome_cossim_topn` function will be removed and (partially) replaced with `sp_matmul_topn`."
        " See the migration guide at 'https://github.com/ing-bank/sparse_dot_topn#readme'."
    )
    if test_nnz_max is not None:
        raise DeprecationWarning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    n_threads = n_jobs if use_threads is True else None
    C = sp_matmul_topn(A=A, B=B, top_n=ntop, threshold=lower_bound, sort=True, n_threads=n_threads)
    if return_best_ntop:
        return C, np.diff(C.indptr).max()
    return C


def sp_matmul(
    A: csr_matrix | csc_matrix | coo_matrix,
    B: csr_matrix | csc_matrix | coo_matrix,
    n_threads: int | None = None,
    idx_dtype: DTypeLike | None = None,
) -> csr_matrix:
    """Compute A * B whilst only storing the `top_n` elements.

    This functions allows large matrices to multiplied with a limited memory footprint.

    Args:
        A: LHS of the multiplication, the number of columns of A determines the orientation of B.
            `A` must be have an {32, 64}bit {int, float} dtype that is of the same kind as `B`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        B: RHS of the multiplication, the number of rows of B must match the number of columns of A or the shape of B.T should be match A.
            `B` must be have an {32, 64}bit {int, float} dtype that is of the same kind as `A`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        n_threads: number of threads to use, `None` implies sequential processing, -1 will use all but one of the available cores.
        idx_dtype: dtype to use for the indices, defaults to 32bit integers

    Throws:
        TypeError: when A, B are not trivially convertable to a `CSR matrix`

    Returns:
        C: result matrix

    """
    idx_dtype = assert_idx_dtype(idx_dtype)
    n_threads: int = n_threads or 1
    if n_threads < 0:
        n_threads = _N_CORES

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

    # basic check. if A or B are all zeros matrix, return all zero matrix directly
    if A.indices.size == 0 or B.indices.size == 0:
        C_indptr = np.zeros(A_nrows + 1, dtype=idx_dtype)
        C_indices = np.zeros(1, dtype=idx_dtype)
        C_data = np.zeros(1, dtype=A.dtype)
        return csr_matrix((C_data, C_indices, C_indptr), shape=(A_nrows, B_ncols))

    kwargs = {
        "nrows": A_nrows,
        "ncols": B_ncols,
        "A_data": A.data,
        "A_indptr": A.indptr if idx_dtype is None else A.indptr.astype(idx_dtype),
        "A_indices": A.indices if idx_dtype is None else A.indices.astype(idx_dtype),
        "B_data": B.data,
        "B_indptr": B.indptr if idx_dtype is None else B.indptr.astype(idx_dtype),
        "B_indices": B.indices if idx_dtype is None else B.indices.astype(idx_dtype),
    }

    func = _core.sp_matmul
    if n_threads > 1:
        if _core._has_openmp_support:
            kwargs["n_threads"] = n_threads
            func = _core.sp_matmul_mt
        else:
            msg = "sparse_dot_topn: extension was compiled without parallelisation (OpenMP) support, ignoring ``n_threads``"
            warnings.warn(msg, stacklevel=1)
    return csr_matrix(func(**kwargs), shape=(A_nrows, B_ncols))


def sp_matmul_topn(
    A: csr_matrix | csc_matrix | coo_matrix,
    B: csr_matrix | csc_matrix | coo_matrix,
    top_n: int,
    threshold: int | float | None = None,
    sort: bool = False,
    density: float | None = None,
    n_threads: int | None = None,
    idx_dtype: DTypeLike | None = None,
) -> csr_matrix:
    """Compute A * B whilst only storing the `top_n` elements.

    This functions allows large matrices to multiplied with a limited memory footprint.

    Args:
        A: LHS of the multiplication, the number of columns of A determines the orientation of B.
            `A` must be have an {32, 64}bit {int, float} dtype that is of the same kind as `B`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        B: RHS of the multiplication, the number of rows of B must match the number of columns of A or the shape of B.T should be match A.
            `B` must be have an {32, 64}bit {int, float} dtype that is of the same kind as `A`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        top_n: the number of results to retain
        sort: return C in a format where the first non-zero element of each row is the largest value
        threshold: only return values greater than the threshold
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
    if n_threads < 0:
        n_threads = _N_CORES
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

    if B_ncols == top_n and (sort is False) and (threshold is None):
        return sp_matmul(A, B, n_threads)

    assert_supported_dtype(A)
    assert_supported_dtype(B)
    ensure_compatible_dtype(A, B)

    # guard against top_n larger than number of cols
    top_n = min(top_n, B_ncols)

    # handle threshold
    if threshold is not None:
        threshold = int(np.rint(threshold)) if np.issubdtype(A.data.dtype, np.integer) else float(threshold)

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
        "B_indices": B.indices if idx_dtype is None else B.indices.astype(idx_dtype),
    }

    func = _core.sp_matmul_topn if not sort else _core.sp_matmul_topn_sorted
    if n_threads > 1:
        if _core._has_openmp_support:
            kwargs["n_threads"] = n_threads
            kwargs.pop("density")
            func = _core.sp_matmul_topn_mt if not sort else _core.sp_matmul_topn_sorted_mt
        else:
            msg = "sparse_dot_topn: extension was compiled without parallelisation (OpenMP) support, ignoring ``n_threads``"
            warnings.warn(msg, stacklevel=1)
    return csr_matrix(func(**kwargs), shape=(A_nrows, B_ncols))


def zip_sp_matmul_topn(top_n: int, C_mats: list[csr_matrix]) -> csr_matrix:
    """Compute zip-matrix C = zip_i C_i = zip_i A * B_i = A * B whilst only storing the `top_n` elements.

    Combine the sub-matrices together and keep only the `top_n` elements per row.

    Pre-calling this function, matrix B has been split row-wise into chunks B_i, and C_i = A * B_i have been calculated.
    This function computes C = zip_i C_i, which is equivalent to A * B when only keeping the `top_n` elements.
    It allows very large matrices to be split and multiplied with a limited memory footprint.

    Args:
        top_n: the number of results to retain; should be smaller or equal to top_n used to obtain C_mats.
        C_mats: a list with each C_i sub-matrix, with format csr_matrix.

    Returns:
        C: zipped result matrix

    Raises:
        TypeError: when not all elements of `C_mats` is a csr_matrix or trivially convertable
        ValueError: when not all elements of `C_mats` has the same number of rows
    """
    _nrows = []
    ncols = []
    data = []
    indptr = []
    indices = []
    for C in C_mats:
        # check correct type of each C
        if isinstance(C, (coo_matrix, csc_matrix)):
            C = C.tocsr(False)
        elif not isinstance(C, csr_matrix):
            msg = f"type of `C` must be one of `csr_matrix`, `csc_matrix` or `csr_matrix`, got `{type(C)}`"
            raise TypeError(msg)

        nrows, c_nc = C.shape
        _nrows.append(nrows)
        ncols.append(c_nc)
        data.append(C.data)
        indptr.append(C.indptr)
        indices.append(C.indices)

    ncols = np.asarray(ncols, int)
    total_cols = ncols.sum()
    if not np.all(np.diff(_nrows) == 0):
        msg = "Each `C` in `C_mats` should have the same number of rows."
        raise ValueError(msg)

    return csr_matrix(
        _core.zip_sp_matmul_topn(
            top_n=top_n, Z_max_nnz=nrows * top_n, nrows=nrows, B_ncols=ncols, data=data, indptr=indptr, indices=indices
        ),
        shape=(nrows, total_cols),
    )
