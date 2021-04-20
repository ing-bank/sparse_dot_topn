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
        ntop_is_optional=False,
        mem_manager_is_C=False
    ):
    """
    This function will return a matrix C in CSR format, where
    C = [sorted top n results and results > lower_bound for each row of A * B]

    Input:
        A and B: two CSR matrix
        ntop: n top results
        lower_bound: a threshold that the element of A*B must greater than
        use_threads: use multi-thread or not
        n_jobs: number of thread, must be >= 1
        ntop_is_optional: if True (default) memory management will be handed 
        over to C/C++ if the first attempt an allocating memory fails, otherwise not
        mem_manager_is_C: (this is mainly for testing purposes) if True, will force
        memory management to be handed over to C/C++. Should be used only when 
        ntop >= number of columns of B

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
        return csr_matrix((data, indices, indptr), shape=(M, N)), 0

    # filled matrices from here on
    n_max_matches_arr = np.full(1, 0, dtype=idx_dtype)
    indptr = np.empty(M+1, dtype=idx_dtype)
    try:
        indices = np.empty(nnz_max, dtype=idx_dtype)
        data = np.empty(nnz_max, dtype=A.dtype)
        if mem_manager_is_C: raise Exception    # This is mainly for testing purposes
    except:
        # if mem_manager_is_C: print('Exception raised! Continuing ...', flush=True)
        if ntop_is_optional or ntop >= N:
        # It is likely you are here because nnz_max is too large. But don't give up just yet! 
        # string_grouper will hand over the memory allocation/management to C++.  C++ will
        # grow the memory allocations for these arrays as needed without any need for nnz_max.
        # Note that reallocations could occur causing data to be copied to other locations 
        # in memory thus impacting performance
            indices = np.empty(0, dtype=idx_dtype)
            data = np.empty(0, dtype=A.dtype)
            if not use_threads:
    
                indices, data, n_max_matches = ct.sparse_dot_free(
                    M, N, np.asarray(A.indptr, dtype=idx_dtype),
                    np.asarray(A.indices, dtype=idx_dtype),
                    A.data,
                    np.asarray(B.indptr, dtype=idx_dtype),
                    np.asarray(B.indices, dtype=idx_dtype),
                    B.data,
                    lower_bound,
                    indptr)
                
            else:
    
                indices, data, n_max_matches = ct_thread.sparse_dot_free_threaded(
                    M, N, np.asarray(A.indptr, dtype=idx_dtype),
                    np.asarray(A.indices, dtype=idx_dtype),
                    A.data,
                    np.asarray(B.indptr, dtype=idx_dtype),
                    np.asarray(B.indices, dtype=idx_dtype),
                    B.data,
                    lower_bound,
                    indptr, n_jobs)
        else:
            raise Exception('Not enough memory!  Data array is too large. Try reducing the value of the\n'
                            'kwarg n_max_matches or do not set it at all.\n')
            
        return csr_matrix((data, indices, indptr), shape=(M, N)), n_max_matches
    
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
            indptr, indices, data, n_max_matches_arr)

    else:
        if n_jobs < 1:
            err_str = 'You select the multi-thread mode and n_job must be a value greater equal than 1!'
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
            indptr, indices, data, n_max_matches_arr, n_jobs)

    return csr_matrix((data, indices, indptr), shape=(M, N)), n_max_matches_arr[0]


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
