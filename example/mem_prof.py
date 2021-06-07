"""
mem_prof.py
This script should be run using the mprof executable (from the
package 'memory_profiler').  For example,
mprof run -T 0.001 -C example\mem_prof.py
mprof plot
"""

from __future__ import print_function
import timeit
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sparse_dot_topn import awesome_cossim_topn  # noqa: F401
# from memory_profiler import profile


@profile
def awesome_cossim_topn_0_threads(a, b, N, thresh):
    return awesome_cossim_topn(a, b, N, thresh, False, 1, True)


@profile
def awesome_cossim_topn_1_thread(a, b, N, thresh):
    return awesome_cossim_topn(a, b, N, thresh, True, 1, True)


@profile
def awesome_cossim_topn_2_threads(a, b, N, thresh):
    return awesome_cossim_topn(a, b, N, thresh, True, 2, True)


@profile
def awesome_cossim_topn_3_threads(a, b, N, thresh):
    return awesome_cossim_topn(a, b, N, thresh, True, 3, True)


@profile
def awesome_cossim_topn_4_threads(a, b, N, thresh):
    return awesome_cossim_topn(a, b, N, thresh, True, 4, True)


@profile
def awesome_cossim_topn_5_threads(a, b, N, thresh):
    return awesome_cossim_topn(a, b, N, thresh, True, 5, True)


@profile
def awesome_cossim_topn_6_threads(a, b, N, thresh):
    return awesome_cossim_topn(a, b, N, thresh, True, 6, True)


@profile
def awesome_cossim_topn_7_threads(a, b, N, thresh):
    return awesome_cossim_topn(a, b, N, thresh, True, 7, True)


N = 4000
thresh = 0.01

nr_vocab = int(26**3)
density = 30/nr_vocab
n_samples = 1000000
n_duplicates = N
nnz_a = int(n_samples * nr_vocab * density)
nnz_b = int(n_duplicates * nr_vocab * density)

print(f'ntop = {N}', flush=True)
print(f'threshold = {thresh}', flush=True)
print(f'density = {density}', flush=True)
print(f'nr_vocab = {nr_vocab}', flush=True)
print(f'n_samples = {n_samples}', flush=True)
print(f'n_duplicates = {n_duplicates}', flush=True)
print(f'nnz_A = {nnz_a}', flush=True)
print(f'nnz_B = {nnz_b}', flush=True)
print('', flush=True)

rng1 = np.random.RandomState(42)

n_matrix_pairs = 2**0
for it in range(n_matrix_pairs):
    
    row = rng1.randint(n_samples, size=nnz_a)
    cols = rng1.randint(nr_vocab, size=nnz_a)
    data = rng1.rand(nnz_a)
    
    a_sparse = coo_matrix((data, (row, cols)), shape=(n_samples, nr_vocab))
    a = a_sparse.tocsr()
    # a = a.astype(np.float32)
    
    row = rng1.randint(n_duplicates, size=nnz_b)
    cols = rng1.randint(nr_vocab, size=nnz_b)
    data = rng1.rand(nnz_b)
    
    b_sparse = coo_matrix((data, (row, cols)), shape=(n_duplicates, nr_vocab))
    b = b_sparse.T.tocsr()
    # b = b.astype(np.float32)
    
    # first run without profiling to bring the memory up to the same level
    # for all subsequent profiled runs:
    C, C_ntop = awesome_cossim_topn(a, b, N, thresh, True, 7, True)
    
    print("Sampling non-parallelized sparse_dot_topn function ... ", end='', flush=True)
    C, C_ntop = awesome_cossim_topn_0_threads(a, b, N, thresh)
    print("Finished.");

    print("Sampling threaded function with 1 thread ... ", end='', flush=True)
    C, C_ntop = awesome_cossim_topn_1_thread(a, b, N, thresh)
    print("Finished.");
    
    print("Sampling threaded function with 2 threads ... ", end='', flush=True)
    C, C_ntop = awesome_cossim_topn_2_threads(a, b, N, thresh)
    print("Finished.");

    print("Sampling threaded function with 3 threads ... ", end='', flush=True)
    C, C_ntop = awesome_cossim_topn_3_threads(a, b, N, thresh)
    print("Finished.");

    print("Sampling threaded function with 4 threads ... ", end='', flush=True)
    C, C_ntop = awesome_cossim_topn_4_threads(a, b, N, thresh)
    print("Finished.");

    print("Sampling threaded function with 5 threads ... ", end='', flush=True)
    C, C_ntop = awesome_cossim_topn_5_threads(a, b, N, thresh)
    print("Finished.");

    print("Sampling threaded function with 6 threads ... ", end='', flush=True)
    C, C_ntop = awesome_cossim_topn_6_threads(a, b, N, thresh)
    print("Finished.");

    print("Sampling threaded function with 7 threads ... ", end='', flush=True)
    C, C_ntop = awesome_cossim_topn_7_threads(a, b, N, thresh)
    print("Finished.");

    print(f'nnz(A*B) = {len(C.data)}', flush=True)
    print(f'ntop(A*B) = {C_ntop}', flush=True)
    print('', flush=True)
