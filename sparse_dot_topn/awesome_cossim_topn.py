import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import isspmatrix_csr
from scipy.sparse import rand

if sys.version_info[0] >= 3:
    from sparse_dot_topn import sparse_dot_topn as ct
else:
    import sparse_dot_topn as ct

def awesome_cossim_topn(A, B, ntop, lower_bound=0):
    """
    This function will return a matrxi C in CSR format, where
    C = [sorted top n results and results > lower_bound for each row of A * B]

    Input:
        A and B: two CSR matrix
        ntop: n top results
        lower_bound: a threshold that the element of A*B must greater than

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

    idx_dtype = np.int32

    nnz_max = M*ntop

    indptr = np.empty(M+1, dtype=idx_dtype)
    indices = np.empty(nnz_max, dtype=idx_dtype)
    data = np.empty(nnz_max, dtype=A.dtype)

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

    return csr_matrix((data,indices,indptr),shape=(M,N))