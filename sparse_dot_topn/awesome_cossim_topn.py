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
        A,
        B,
        ntop,
        lower_bound=0,
        use_threads=False,
        n_jobs=1,
        mem_manager_is_C=False,
        return_best_topn=False
    ):
    """
    This function will return a matrix C in CSR format, where
    C = [sorted top n results > lower_bound for each row of A * B].
    If return_best_topn=True then best_topn
    (the true maximum number of elements > lower_bound per row of A * B)
    will also be returned in a tuple together with C as (C, best_topn).

    Input:
        A and B: two CSR matrices
        ntop: top n results
        lower_bound: a threshold that the element of A*B must be greater than
        use_threads: use multi-thread or not
        n_jobs: number of thread, must be >= 1
        mem_manager_is_C: (default: False) this is mainly for testing purposes. if 
                          True, will force memory management to be handed over to
                          C/C++.
        return_best_topn: (default: False) if True, will return best_topn together 
                          with C as a tuple: (C, best_topn)

    Output:
        C: result matrix (returned alone, if return_best_topn=False)
        best_topn: The true maximum number of elements > lower_bound per row of 
                   A * B returned together with C as a tuple: (C, best_topn). It is 
                   returned only if return_best_topn=True.

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
        if return_best_topn:
            return output, 0
        else:
            return output

    # filled matrices from here on
    indptr = np.empty(M+1, dtype=idx_dtype)
    try:
        indices = np.empty(nnz_max, dtype=idx_dtype)
        data = np.empty(nnz_max, dtype=A.dtype)
        if mem_manager_is_C: raise MemoryError    # This is mainly for testing purposes
    except MemoryError:
        # if mem_manager_is_C: print('Exception raised! Continuing ...', flush=True)
        # It is likely you are here because nnz_max is too large. But don't give up just yet! 
        # sparse_dot_topn will hand over the memory allocation/management to C++.  C++ will
        # grow the memory allocations for these arrays as needed without any need for nnz_max.
        # Note that reallocations could occur causing data to be copied to other locations 
        # in memory thus impacting performance
        indices = np.empty(0, dtype=idx_dtype)
        data = np.empty(0, dtype=A.dtype)
        if not use_threads:

            indices, data, best_topn = ct.sparse_dot_free(
                M, N, np.asarray(A.indptr, dtype=idx_dtype),
                np.asarray(A.indices, dtype=idx_dtype),
                A.data,
                np.asarray(B.indptr, dtype=idx_dtype),
                np.asarray(B.indices, dtype=idx_dtype),
                B.data,
                ntop, lower_bound,
                indptr
            )
            
        else:

            indices, data, best_topn = ct_thread.sparse_dot_free_threaded(
                M, N, np.asarray(A.indptr, dtype=idx_dtype),
                np.asarray(A.indices, dtype=idx_dtype),
                A.data,
                np.asarray(B.indptr, dtype=idx_dtype),
                np.asarray(B.indices, dtype=idx_dtype),
                B.data,
                ntop, lower_bound,
                indptr, n_jobs
            )

    else:
        # no exception was raised; then use old function (as it is expected to be the fastest)
        
        best_topn_arr = np.full(1, 0, dtype=idx_dtype)
        
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
                indptr, indices, data, best_topn_arr
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
                indptr, indices, data, best_topn_arr, n_jobs
            )
        
        best_topn = best_topn_arr[0]
    
    # prepare and return the output:
    output = csr_matrix((data, indices, indptr), shape=(M, N))
    if return_best_topn:
        return output, best_topn
    else:
        return output


def awesome_cossim_only_max_nnz_col(A, B, use_threads=False, n_jobs=1):
    """
    This function will return the maximum number of columns set
    per row over all rows of A * B

    Input:
        A and B: two CSR matrix
        use_threads: use multi-thread or not
        n_jobs: number of thread, must be >= 1

    Output:
        minmax_topn: maximum number of columns set
                     per row over all rows of A * B

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

    minmax_topn = np.full(1, 0, dtype=idx_dtype)

    # basic check. if A or B are all zeros matrix, return 0 directly
    if len(A.indices) == 0 or len(B.indices) == 0:
        return 0

    if not use_threads:

        ct.sparse_dot_only_max_nnz_col(
            M, N,
            np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            minmax_topn)

    else:
        if n_jobs < 1:
            err_str = 'You select the multi-thread mode and n_job must be a value greater equal than 1!'
            raise ValueError(err_str)

        ct_thread.sparse_dot_only_max_nnz_col_threaded(
            M, N,
            np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            minmax_topn, n_jobs)

    return minmax_topn[0]
