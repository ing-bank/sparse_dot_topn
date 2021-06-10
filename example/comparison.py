"""
This file compare our boosting method with calling scipy+numpy function directly
"""

from __future__ import print_function
import timeit
import numpy as np
from scipy.sparse import coo_matrix
from sparse_dot_topn import awesome_cossim_topn  # noqa: F401

N = 1000
thresh = 0.01

nr_vocab = 2 << 24
density = 1e-6
n_samples = 1000000
n_duplicates = 1000000
nnz_a = int(n_samples * nr_vocab * density)
nnz_b = int(n_duplicates * nr_vocab * density)


print(f'density = {density}', flush=True)
print(f'nr_vocab = {nr_vocab}', flush=True)
print(f'n_samples = {n_samples}', flush=True)
print(f'n_duplicates = {n_duplicates}', flush=True)
print(f'nnz_a = {nnz_a}', flush=True)
print(f'nnz_b = {nnz_b}', flush=True)
print('\n', flush=True)

rng1 = np.random.RandomState(42)
rng2 = np.random.RandomState(43)

row = rng1.randint(n_samples, size=nnz_a)
cols = rng2.randint(nr_vocab, size=nnz_a)
data = rng1.rand(nnz_a)
dtype = np.float32

a_sparse = coo_matrix((data, (row, cols)), shape=(n_samples, nr_vocab), dtype=dtype)
a = a_sparse.tocsr()

row = rng1.randint(n_duplicates, size=nnz_b)
cols = rng2.randint(nr_vocab, size=nnz_b)
data = rng1.rand(nnz_b)

b_sparse = coo_matrix((data, (row, cols)), shape=(n_duplicates, nr_vocab), dtype=dtype)
b = b_sparse.T.tocsr()


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

print("Threaded function with 3 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 3)',
                    number=3,
                    globals=globals())
print(rtv)

print("Threaded function with 4 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 4)',
                    number=3,
                    globals=globals())
print(rtv)

print("Threaded function with 5 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 5)',
                    number=3,
                    globals=globals())
print(rtv)

print("Threaded function with 6 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 6)',
                    number=3,
                    globals=globals())
print(rtv)

print("Threaded function with 7 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 7)',
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
