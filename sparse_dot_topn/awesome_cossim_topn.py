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


def awesome_cossim_topn(
        A, B, ntop, lower_bound=0, use_threads=False, n_jobs=1, scout_nnz=False, return_best_ntop=False):
    """
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
        scout_nnz: (default: False) this is mainly for testing purposes. if 
                   True, will force a memory-size determination before computing
                   the results.
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
        output = csr_matrix((data, indices, indptr), shape=(M, N))
        if return_best_ntop:
            return output, 0
        else:
            return output

    # filled matrices from here on
    indptr = np.empty(M+1, dtype=idx_dtype)
    try:
        indices = np.empty(nnz_max, dtype=idx_dtype)
        data = np.empty(nnz_max, dtype=A.dtype)
        if scout_nnz: raise MemoryError    # This is mainly for testing purposes
    except MemoryError:
        # if scout_nnz: print('Exception raised! Continuing ...', flush=True)
        # It is likely you are here because nnz_max is too large. But don't give up just yet! 
        # sparse_dot_topn will go ahead and count the exact amount of memory required.
        if not use_threads:
            
            nnz = ct.sparse_dot_only_nnz(M, N, np.asarray(A.indptr, dtype=idx_dtype),
                np.asarray(A.indices, dtype=idx_dtype),
                A.data,
                np.asarray(B.indptr, dtype=idx_dtype),
                np.asarray(B.indices, dtype=idx_dtype),
                B.data,
                ntop, lower_bound
            )
            
        else:

            nnz = ct_thread.sparse_dot_only_nnz_threaded(
                M, N, np.asarray(A.indptr, dtype=idx_dtype),
                np.asarray(A.indices, dtype=idx_dtype),
                A.data,
                np.asarray(B.indptr, dtype=idx_dtype),
                np.asarray(B.indices, dtype=idx_dtype),
                B.data,
                ntop, lower_bound, n_jobs
            )
            
        nnz = max(1, nnz)
        indices = np.empty(nnz, dtype=idx_dtype)
        data = np.empty(nnz, dtype=A.dtype)
        
    # no exception was raised; then use old function (as it is expected to be the fastest)
    
    best_ntop_arr = np.full(1, 0, dtype=idx_dtype)
    
    if not use_threads:
    
        ct.sparse_dot_topn_extd(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data, best_ntop_arr
        )

    else:
        if n_jobs < 1:
            err_str = 'Whenever you select the multi-thread mode, n_job must be greater than or equal to 1!'
            raise ValueError(err_str)

        ct_thread.sparse_dot_topn_extd_threaded(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data, best_ntop_arr, n_jobs
        )
    
    # prepare and return the output:
    output = csr_matrix((data, indices, indptr), shape=(M, N))
    if return_best_ntop:
        return output, best_ntop_arr[0]
    else:
        return output

