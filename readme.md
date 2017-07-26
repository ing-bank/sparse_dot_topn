# sparse\_dot\_topn
Comparing very large feature vectors and picking the best matches, in practice often results in performing a sparse matrix multiplication followed by selecting the top-n multiplication results. In this package, we implement a customized Cython function for this purpose. When comparing our Cythonic approach to doing the same use with SciPy and NumPy functions, our approach improves the speed by about 40% and reduces memory consumption. The GitHub code of our approach is available here.

This package is made by ING Wholesale Banking Advanced Analytics team. This [blog](https://medium.com/@ingwbaa/https-medium-com-ingwbaa-boosting-selection-of-the-most-similar-entities-in-large-scale-datasets-450b3242e618) explains how we implement it.

## Example
``` python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import rand
import sparse_dot_topn.sparse_dot_topn as ct

def awesome_cossim_top(A, B, ntop, lower_bound=0):
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

a = rand(4, 100000, density=0.01, format='csr')
b = rand(100000, 10, density=0.01, format='csr')

# top 5 results per row which element is greater than 2
c = awesome_cossim_top(a, b, 5, 2)
```

## Install
``` sh
make install
```
or
``` python
python setup.py install
```

## uninstall
``` sh
make remove
```