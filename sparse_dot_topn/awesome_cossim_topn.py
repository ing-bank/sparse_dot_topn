import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import isspmatrix_csr
import psutil

if sys.version_info[0] >= 3:
    from sparse_dot_topn import sparse_dot_topn as ct
    from sparse_dot_topn import sparse_dot_topn_threaded as ct_thread
else:
    import sparse_dot_topn as ct
    import sparse_dot_topn_threaded as ct_thread


_IDX_DTYPE = np.int32


def awesome_cossim_topn(
    A,
    B,
    ntop,
    lower_bound=0,
    use_threads=False,
    n_jobs=1,
    return_best_ntop=False,
    test_nnz_max=-1,
):
    """Compute top-n dot product of two sparse matrices.

    If `A` and `B` are L2 normalised TF-IFD matrices this is their cosine similarity.

    This function will return a matrix C in CSR format, where
    C = [sorted top n results > lower_bound for each row of A * B].
    If return_best_ntop=True then best_ntop
    (the true maximum number of elements > lower_bound per row of A * B)
    will also be returned in a tuple together with C as (C, best_ntop).

    Input:
        A and B: two CSR matrices
        ntop: top n results
        lower_bound: a threshold that the element of A*B must be greater than
        use_threads: use multi-thread or not
        n_jobs: number of thread, must be >= 1
        return_best_ntop: (default: False) if True, will return best_ntop together
                          with C as a tuple: (C, best_ntop)

    Output:
        C: result matrix (returned alone, if return_best_ntop=False)
        best_ntop: The true maximum number of elements > lower_bound per row of
                   A * B returned together with C as a tuple: (C, best_ntop). It is
                   returned only if return_best_ntop=True.

    N.B. if A and B are not in CSR format, they will be converted to CSR
    """

    if not isspmatrix_csr(A):
        A = A.tocsr()
    if not isspmatrix_csr(B):
        B = B.tocsr()

    dtype = A.dtype
    assert B.dtype == dtype
    lower_bound = dtype.type(lower_bound)  # Casting this scalar to the same type

    M, K1 = A.shape
    K2, N = B.shape

    if K1 != K2:
        if N == K1:
            B = B.T
            K2, N = B.shape
        else:
            raise ValueError("`A.shape[1]` must be equal to `B.shape[0]`.")

    # the maximum number of non-zero elements
    nnz_max = M * ntop

    # basic check. if A or B are all zeros matrix, return all zero matrix directly
    if len(A.indices) == 0 or len(B.indices) == 0:
        indptr = np.zeros(M + 1, dtype=_IDX_DTYPE)
        indices = np.zeros(nnz_max, dtype=_IDX_DTYPE)
        data = np.zeros(nnz_max, dtype=A.dtype)
        output = csr_matrix((data, indices, indptr), shape=(M, N))
        if return_best_ntop:
            return output, 0
        else:
            return output

    required_bytes = (np.dtype(_IDX_DTYPE).itemsize + A.itemsize) * nnz_max
    available_bytes = psutil.virtual_memory().available
    if required_bytes > available_bytes:
        raise MemoryError(
            f"the maximum required memory {required_bytes / 1000}kB is"
            f"larger than the available memory {available_bytes / 1000} kB"
        )

    # filled matrices from here on
    indptr = np.empty(M + 1, dtype=_IDX_DTYPE)
    indices = np.empty(nnz_max, dtype=_IDX_DTYPE)
    data = np.empty(nnz_max, dtype=A.dtype)
    best_ntop_arr = np.zeros(1, dtype=_IDX_DTYPE)

    if not use_threads:
        alt_indices, alt_data = ct.sparse_dot_topn_extd(
            M,
            N,
            np.asarray(A.indptr, dtype=_IDX_DTYPE),
            np.asarray(A.indices, dtype=_IDX_DTYPE),
            A.data,
            np.asarray(B.indptr, dtype=_IDX_DTYPE),
            np.asarray(B.indices, dtype=_IDX_DTYPE),
            B.data,
            ntop,
            lower_bound,
            indptr,
            indices,
            data,
            best_ntop_arr,
        )

    else:
        if n_jobs < 1:
            raise ValueError("if `use_threads` is true than `n_job` must be >= 1.")

        alt_indices, alt_data = ct_thread.sparse_dot_topn_extd_threaded(
            M,
            N,
            np.asarray(A.indptr, dtype=_IDX_DTYPE),
            np.asarray(A.indices, dtype=_IDX_DTYPE),
            A.data,
            np.asarray(B.indptr, dtype=_IDX_DTYPE),
            np.asarray(B.indices, dtype=_IDX_DTYPE),
            B.data,
            ntop,
            lower_bound,
            indptr,
            indices,
            data,
            best_ntop_arr,
            n_jobs,
        )

    if alt_indices is not None:
        indices = alt_indices
        data = alt_data

    # prepare and return the output:
    output = csr_matrix((data, indices, indptr), shape=(M, N))
    if return_best_ntop:
        return output, best_ntop_arr[0]
    else:
        return output
