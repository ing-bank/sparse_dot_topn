# Copyright (c) 2023 ING Analytics Wholesale Banking
from __future__ import annotations

import psutil
from scipy.sparse import csr_matrix

from sparse_dot_topn.lib import _sparse_dot_topn_core as _core

__all__ = ["sp_matmul_topn"]

_N_CORES = psutil.cpu_count(logical=False) - 1


def sp_matmul_topn(
    A: csr_matrix, B: csr_matrix, top_n: int, sort: bool, threshold: int | float, density: float
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
        sort: return C in a format where the first non-zero element of each row is the largest value
        threshold: only return values greater than the threshold, by default this 0.0
        density: the expected density of the result considering `top_n`. The expected number of non-zero elements
            in C should <= (`density` * `top_n` * `A.shape[0]`) otherwise the memory has to reallocated.
            This value should only be set if you have a strong expectation as being wrong incurs a realloaction penalty.

    Returns:
        C: result matrix

    """
    nrows = A.shape[0]
    ncols = B.shape[1]
    func = _core.sp_matmul_topn if not sort else _core.sp_matmul_topn_sorted
    return csr_matrix(
        func(top_n, nrows, ncols, threshold, density, A.data, A.indptr, A.indices, B.data, B.indptr, B.indices),
        shape=(nrows, ncols),
    )


def sp_matmul_topn_mt(
    A: csr_matrix, B: csr_matrix, top_n: int, sort: bool, threshold: int | float, n_threads: int
) -> csr_matrix:
    """Compute A * B whilst only storing the `top_n` elements using multiprocessing.

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
        sort: return C in a format where the first non-zero element of each row is the largest value
        threshold: only return values greater than the threshold, by default this 0.0
        n_threads: number of threads to use, -1 will use all but one of the available cores.

    Returns:
        C: result matrix

    """
    nrows = A.shape[0]
    ncols = B.shape[1]
    n_threads = n_threads if n_threads > 0 else _N_CORES
    func = _core.sp_matmul_topn_mt if not sort else _core.sp_matmul_topn_sorted_mt
    return csr_matrix(
        func(top_n, nrows, ncols, threshold, n_threads, A.data, A.indptr, A.indices, B.data, B.indptr, B.indices),
        shape=(nrows, ncols),
    )
