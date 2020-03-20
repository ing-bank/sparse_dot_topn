import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import isspmatrix_csr

if sys.version_info[0] >= 3:
    from sparse_dot_topn import sparse_dot_topn as ct
    from sparse_dot_topn import sparse_dot_topn_threaded as ct_thread
else:
    import sparse_dot_topn as ct
    import sparse_dot_topn_threaded as ct_thread


def awesome_cossim_topn(A, B, ntop, lower_bound=0, use_threads=False, n_jobs=1):
    """
    This function will return a matrxi C in CSR format, where
    C = [sorted top n results and results > lower_bound for each row of A * B]

    Input:
        A and B: two CSR matrix
        ntop: n top results
        lower_bound: a threshold that the element of A*B must greater than
        use_threads: use multi-thread or not
        n_jobs: number of thread, must be >= 1

    Output:
        C: result matrix

    N.B. if A and B are not CSR format, they will be converted to CSR
    """
    if not isspmatrix_csr(A):
        A = A.tocsr()

    if not isspmatrix_csr(B):
        B = B.tocsr()

    M, K1 = A.shape
    K2, N = B.shape

    if K1 != K2:
        err_str = 'A matrix multiplication will be operated. A.shape[1] must be equal to B.shape[0]!'
        raise ValueError(err_str)

    idx_dtype = np.int32

    nnz_max = M*ntop

    # basic check. if A or B are all zeros matrix, return all zero matrix directly
    if len(A.indices) == 0 or len(B.indices) == 0:
        indptr = np.zeros(M + 1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)
        return csr_matrix((data, indices, indptr), shape=(M, N))

    # filled matrices from here on
    indptr = np.empty(M+1, dtype=idx_dtype)
    indices = np.empty(nnz_max, dtype=idx_dtype)
    data = np.empty(nnz_max, dtype=A.dtype)

    if not use_threads:

        ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)

    else:
        if n_jobs < 1:
            err_str = 'You select the multi-thread mode and n_job must be a value greater equal than 1!'
            raise ValueError(err_str)

        ct_thread.sparse_dot_topn_threaded(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data, n_jobs)

    return csr_matrix((data, indices, indptr), shape=(M, N))
