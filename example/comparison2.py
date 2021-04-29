"""
This file compare our boosting method with calling scipy+numpy function directly
"""

from __future__ import print_function
import timeit
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sparse_dot_topn import awesome_cossim_topn  # noqa: F401

df = pd.DataFrame(columns=['sample', '#threads', 'python', '+scout', '%inc'])

N = 1000
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
rng2 = np.random.RandomState(43)

n_matrix_pairs = 2**4
nnz_arr = np.full(n_matrix_pairs, 0)
ntop_arr = np.full(n_matrix_pairs, 0)
r = 0
for it in range(n_matrix_pairs):
    
    row = rng1.randint(n_samples, size=nnz_a)
    cols = rng2.randint(nr_vocab, size=nnz_a)
    data = rng1.rand(nnz_a)
    
    a_sparse = coo_matrix((data, (row, cols)), shape=(n_samples, nr_vocab))
    a = a_sparse.tocsr()
    
    row = rng1.randint(n_duplicates, size=nnz_b)
    cols = rng2.randint(nr_vocab, size=nnz_b)
    data = rng1.rand(nnz_b)
    
    b_sparse = coo_matrix((data, (row, cols)), shape=(n_duplicates, nr_vocab))
    b = b_sparse.T.tocsr()
    
    C, C_ntop = awesome_cossim_topn(a, b, N, thresh, return_best_ntop=True)
    print(f'nnz(A*B) = {len(C.data)}', flush=True)
    print(f'ntop(A*B) = {C_ntop}', flush=True)
    print('', flush=True)
    nnz_arr[it] = len(C.data)
    ntop_arr[it] = C_ntop
    
    
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
