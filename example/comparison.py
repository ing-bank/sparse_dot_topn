"""
This file compare our boosting method with calling scipy+numpy function directly
"""

from __future__ import print_function
import timeit
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import rand
from sparse_dot_topn import awesome_cossim_topn

N = 10
thresh = 0.01

a = rand(100, 1000000, density=0.005, format='csr')
b = rand(1000000, 200, density=0.005, format='csr')

# top 5 results per row

print("Original sparse_dot_topn function")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh)',
                    number=3,
                    globals=globals())
print(rtv)

print("Threaded function with 1 thread")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 1)',
                    number=3,
                    globals=globals())
print(rtv)

print("Threaded function with 2 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 2)',
                    number=3,
                    globals=globals())
print(rtv)

# use scipy and numpy function


def get_csr_ntop_idx_data(csr_row, ntop):
    """
    Get list (row index, score) of the n top matches
    """
    nnz = csr_row.getnnz()
    if nnz == 0:
        return None
    elif nnz <= ntop:
        result = zip(csr_row.indices, csr_row.data)
    else:
        arg_idx = np.argpartition(csr_row.data, -ntop)[-ntop:]
        result = zip(csr_row.indices[arg_idx], csr_row.data[arg_idx])

    return sorted(result, key=lambda x: -x[1])


def scipy_cossim_top(A, B, ntop, lower_bound=0):
    C = A.dot(B)
    return [get_csr_ntop_idx_data(row, ntop) for row in C]

# top 5 results per row which element is greater than 2


print("Scipy+numpy original function")

rtv = timeit.timeit('scipy_cossim_top(a, b, N, thresh)',
                    number=3,
                    globals=globals())
print(rtv)
