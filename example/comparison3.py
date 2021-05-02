"""
This file compare our boosting method with calling scipy+numpy function directly
"""

from __future__ import print_function
import timeit
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sparse_dot_topn import awesome_cossim_topn  # noqa: F401

df = pd.DataFrame(columns=['sample', '#threads', 'python', '+scout', '%inc'])

a = load_npz('sparse_matrix_A.npz')
b = load_npz('sparse_matrix_B.npz')

tic = time.perf_counter()
p = np.random.permutation(a.shape[0])
a = a[p]
toc = time.perf_counter()
print(f'shuffle(A) took {(toc - tic):0.4f} seconds', flush=True)


N = b.shape[1]
thresh = 0.8

nr_vocab = b.shape[0]
density_A = len(a.data)/(a.shape[0]*a.shape[1]) 
density_B = len(b.data)/(b.shape[0]*b.shape[1]) 
n_samples = a.shape[0]
n_duplicates = b.shape[1]
nnz_a = len(a.data)
nnz_b = len(b.data)

print(f'ntop = {N}', flush=True)
print(f'threshold = {thresh}', flush=True)
print(f'density(A) = {density_A}', flush=True)
print(f'density(B) = {density_B}', flush=True)
print(f'nr_vocab = {nr_vocab}', flush=True)
print(f'n_samples = {n_samples}', flush=True)
print(f'n_duplicates = {n_duplicates}', flush=True)
print(f'nnz_A = {nnz_a}', flush=True)
print(f'nnz_B = {nnz_b}', flush=True)
print('', flush=True)

n_matrix_pairs = 1
nnz_arr = np.full(n_matrix_pairs, 0)
ntop_arr = np.full(n_matrix_pairs, 0)
r = 0
it = 0

tic = time.perf_counter()
C, C_ntop = awesome_cossim_topn(a, b, N, thresh, scout_nnz=True, use_threads=True, n_jobs = 7, return_best_ntop=True)
toc = time.perf_counter()

print('scout_nnz=True, use_threads=True, n_jobs = 7')
print(f'nnz(A*B) = {len(C.data)}', flush=True)
print(f'ntop(A*B) = {C_ntop}', flush=True)
print(f'duration(A*B) = {(toc - tic):0.4f}', flush=True)

tic = time.perf_counter()
C = C[np.argsort(p)]
toc = time.perf_counter()
print(f'unshuffle(A*B) took {(toc - tic):0.4f} seconds', flush=True)


tic = time.perf_counter()
C, C_ntop = awesome_cossim_topn(a, b, N, thresh, use_threads=True, n_jobs = 7, return_best_ntop=True)
toc = time.perf_counter()

print('scout_nnz=False, use_threads=True, n_jobs = 7')
print(f'nnz(A*B) = {len(C.data)}', flush=True)
print(f'ntop(A*B) = {C_ntop}', flush=True)
print(f'duration(A*B) = {(toc - tic):0.4f}', flush=True)

print('', flush=True)
nnz_arr[it] = len(C.data)
ntop_arr[it] = C_ntop
del C
del C_ntop

# top 5 results per row

print("Non-parallelized sparse_dot_topn function")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh)',
                    number=3,
                    globals=globals())
rtv2 = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, scout_nnz=True)',
                    number=3,
                    globals=globals())
df.loc[r] = [it, 0, rtv, rtv2, 100.*(rtv2 - rtv)/rtv]
r += 1
print('sample\t\tpython\t\t+scout', flush=True)
print(f'{it}\t\t{rtv:7.4f}\t\t{rtv2:7.4f}', flush=True)

print("Threaded function with 1 thread")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 1)',
                    number=3,
                    globals=globals())
rtv2 = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 1, scout_nnz=True)',
                    number=3,
                    globals=globals())
df.loc[r] = [it, 1, rtv, rtv2, 100.*(rtv2 - rtv)/rtv]
r += 1
print('sample\t\tpython\t\t+scout', flush=True)
print(f'{it}\t\t{rtv:7.4f}\t\t{rtv2:7.4f}', flush=True)

print("Threaded function with 2 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 2)',
                    number=3,
                    globals=globals())
rtv2 = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 2, scout_nnz=True)',
                    number=3,
                    globals=globals())
df.loc[r] = [it, 2, rtv, rtv2, 100.*(rtv2 - rtv)/rtv]
r += 1
print('sample\t\tpython\t\t+scout', flush=True)
print(f'{it}\t\t{rtv:7.4f}\t\t{rtv2:7.4f}', flush=True)

print("Threaded function with 3 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 3)',
                    number=3,
                    globals=globals())
rtv2 = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 3, scout_nnz=True)',
                    number=3,
                    globals=globals())
df.loc[r] = [it, 3, rtv, rtv2, 100.*(rtv2 - rtv)/rtv]
r += 1
print('sample\t\tpython\t\t+scout', flush=True)
print(f'{it}\t\t{rtv:7.4f}\t\t{rtv2:7.4f}', flush=True)

print("Threaded function with 4 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 4)',
                    number=3,
                    globals=globals())
rtv2 = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 4, scout_nnz=True)',
                    number=3,
                    globals=globals())
df.loc[r] = [it, 4, rtv, rtv2, 100.*(rtv2 - rtv)/rtv]
r += 1
print('sample\t\tpython\t\t+scout', flush=True)
print(f'{it}\t\t{rtv:7.4f}\t\t{rtv2:7.4f}', flush=True)

print("Threaded function with 5 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 5)',
                    number=3,
                    globals=globals())
rtv2 = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 5, scout_nnz=True)',
                    number=3,
                    globals=globals())
df.loc[r] = [it, 5, rtv, rtv2, 100.*(rtv2 - rtv)/rtv]
r += 1
print('sample\t\tpython\t\t+scout', flush=True)
print(f'{it}\t\t{rtv:7.4f}\t\t{rtv2:7.4f}', flush=True)

print("Threaded function with 6 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 6)',
                    number=3,
                    globals=globals())
rtv2 = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 6, scout_nnz=True)',
                    number=3,
                    globals=globals())
df.loc[r] = [it, 6, rtv, rtv2, 100.*(rtv2 - rtv)/rtv]
r += 1
print('sample\t\tpython\t\t+scout', flush=True)
print(f'{it}\t\t{rtv:7.4f}\t\t{rtv2:7.4f}', flush=True)

print("Threaded function with 7 threads")

rtv = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 7)',
                    number=3,
                    globals=globals())
rtv2 = timeit.timeit('awesome_cossim_topn(a, b, N, thresh, True, 7, scout_nnz=True)',
                    number=3,
                    globals=globals())
df.loc[r] = [it, 7, rtv, rtv2, 100.*(rtv2 - rtv)/rtv]
r += 1
print('sample\t\tpython\t\t+scout', flush=True)
print(f'{it}\t\t{rtv:7.4f}\t\t{rtv2:7.4f}', flush=True)

print('')
print(f'nnz(A*B) = {nnz_arr[:(it + 1)].mean()} +/- {nnz_arr[:(it + 1)].std()}')
print(f'ntop(A*B) = {ntop_arr[:(it + 1)].mean()} +/- {ntop_arr[:(it + 1)].std()}')
print('')
df = df.astype({
    'sample': np.int64, '#threads': np.int64, 'python': np.float64, '+scout': np.float64, '%inc': np.float64})
results = df.groupby('#threads', as_index=True, sort=True)[['python', '+scout', '%inc']].mean()

print(results)
print('')
print('')
