# -*- coding: utf-8 -*-

from sparse_dot_topn import awesome_cossim_topn
from scipy.sparse.csr import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import rand
import numpy as np
import pandas as pd
import multiprocessing
import pytest

PRUNE_THRESHOLD = 0.1
NUM_CANDIDATES = 3
MEM_MANAGER_IS_C = True
USE_THREADS = True
MAX_N_PROCESSES = min(8, multiprocessing.cpu_count()) - 1


def get_n_top_sparse(mat, n_top=10):
    """
    Get list of (index, value) of the n largest elements in a 1-dimensional sparse matrix

    :param mat: input sparse matrix
    :param n_top: number of largest elements, default is 10.
    :return: sorted list of largest elements
    """
    length = mat.getnnz()
    if length == 0:
        return None
    if length <= n_top:
        result = list(zip(mat.indices, mat.data))
    else:
        arg_idx = np.argpartition(mat.data, -n_top)[-n_top:]
        result = list(zip(mat.indices[arg_idx], mat.data[arg_idx]))
    return sorted(result, key=lambda x: -x[1])


def helper_awesome_cossim_topn_dense(
        a_dense,
        b_dense,
        mem_manager_is_C=False,
        use_threads=False,
        n_jobs=1
    ):
    dense_result = np.dot(a_dense, np.transpose(b_dense))  # dot product
    sparse_result = csr_matrix(dense_result)
    sparse_result_top3 = [get_n_top_sparse(row, NUM_CANDIDATES)
                          for row in sparse_result]  # get ntop using the old method

    pruned_dense_result = dense_result.copy()
    pruned_dense_result[pruned_dense_result < PRUNE_THRESHOLD] = 0  # prune low similarity
    pruned_sparse_result = csr_matrix(pruned_dense_result)
    pruned_sparse_result_top3 = [get_n_top_sparse(row, NUM_CANDIDATES) for row in pruned_sparse_result]

    a_csr = csr_matrix(a_dense)
    b_csr_t = csr_matrix(b_dense).T

    awesome_result = awesome_cossim_topn(
        a_csr, b_csr_t, len(b_dense),
        0.0,
        mem_manager_is_C=mem_manager_is_C, 
        use_threads=use_threads,
        n_jobs=n_jobs
    )
    awesome_result_top3 = awesome_cossim_topn(
        a_csr,
        b_csr_t,
        NUM_CANDIDATES,
        0.0,
        mem_manager_is_C=mem_manager_is_C,
        use_threads=use_threads,
        n_jobs=n_jobs
    )
    awesome_result_top3 = [list(zip(row.indices, row.data)) if len(
        row.data) > 0 else None for row in awesome_result_top3]  # make comparable, normally not needed

    pruned_awesome_result = awesome_cossim_topn(
        a_csr,
        b_csr_t,
        len(b_dense),
        PRUNE_THRESHOLD,
        mem_manager_is_C=mem_manager_is_C,
        use_threads=use_threads,
        n_jobs=n_jobs
    )
    pruned_awesome_result_top3 = awesome_cossim_topn(
        a_csr,
        b_csr_t,
        NUM_CANDIDATES,
        PRUNE_THRESHOLD,
        mem_manager_is_C=mem_manager_is_C,
        use_threads=use_threads,
        n_jobs=n_jobs
    )
    pruned_awesome_result_top3 = [list(zip(row.indices, row.data)) if len(
        row.data) > 0 else None for row in pruned_awesome_result_top3]

    # no candidate selection, no pruning
    assert awesome_result.nnz == sparse_result.nnz
    # no candidate selection, below PRUNE_THRESHOLD similarity pruned
    assert pruned_awesome_result.nnz == pruned_sparse_result.nnz

    all_none1 = np.all(pd.isnull(awesome_result_top3)) and np.all(pd.isnull(sparse_result_top3))
    all_none2 = np.all(pd.isnull(pruned_awesome_result_top3)) and np.all(pd.isnull(pruned_sparse_result_top3))

    # top NUM_CANDIDATES candidates selected, no pruning
    if not all_none1:
        np.testing.assert_array_almost_equal(awesome_result_top3, sparse_result_top3)
    else:
        assert len(awesome_result_top3) == len(sparse_result_top3)
    # top NUM_CANDIDATES candidates selected, below PRUNE_THRESHOLD similarity pruned
    if not all_none2:
        np.testing.assert_array_almost_equal(pruned_awesome_result_top3, pruned_sparse_result_top3)
    else:
        assert len(pruned_awesome_result_top3) == len(pruned_sparse_result_top3)


def helper_awesome_cossim_topn_sparse(
        a_sparse,
        b_sparse,
        flag=True,
        mem_manager_is_C=False,
        use_threads=False,
        n_jobs=1
    ):
    # Note: helper function using awesome_cossim_topn
    sparse_result = a_sparse.dot(b_sparse.T)  # dot product
    sparse_result_top3 = [get_n_top_sparse(row, NUM_CANDIDATES)
                          for row in sparse_result]  # get ntop using the old method

    pruned_sparse_result = sparse_result.copy()
    pruned_sparse_result[pruned_sparse_result < PRUNE_THRESHOLD] = 0  # prune low similarity
    pruned_sparse_result.eliminate_zeros()
    pruned_sparse_result_top3 = [get_n_top_sparse(row, NUM_CANDIDATES) for row in pruned_sparse_result]

    a_csr = csr_matrix(a_sparse)
    b_csr_t = csr_matrix(b_sparse).T

    awesome_result = awesome_cossim_topn(
        a_csr,
        b_csr_t,
        b_sparse.shape[0],
        0.0,
        mem_manager_is_C=mem_manager_is_C,
        use_threads=use_threads,
        n_jobs=n_jobs
    )
    awesome_result_top3 = awesome_cossim_topn(
        a_csr,
        b_csr_t,
        NUM_CANDIDATES,
        0.0,
        mem_manager_is_C=mem_manager_is_C,
        use_threads=use_threads,
        n_jobs=n_jobs
    )
    awesome_result_top3 = [list(zip(row.indices, row.data)) if len(
        row.data) > 0 else None for row in awesome_result_top3]  # make comparable, normally not needed

    pruned_awesome_result = awesome_cossim_topn(
        a_csr,
        b_csr_t,
        b_sparse.shape[0],
        PRUNE_THRESHOLD,
        mem_manager_is_C=mem_manager_is_C,
        use_threads=use_threads,
        n_jobs=n_jobs
    )
    pruned_awesome_result_top3 = awesome_cossim_topn(
        a_csr,
        b_csr_t,
        NUM_CANDIDATES,
        PRUNE_THRESHOLD,
        mem_manager_is_C=mem_manager_is_C,
        use_threads=use_threads,
        n_jobs=n_jobs
    )
    pruned_awesome_result_top3 = [list(zip(row.indices, row.data)) if len(
        row.data) > 0 else None for row in pruned_awesome_result_top3]

    # no candidate selection, no pruning
    assert awesome_result.nnz == sparse_result.nnz
    # no candidate selection, below PRUNE_THRESHOLD similarity pruned
    assert pruned_awesome_result.nnz == pruned_sparse_result.nnz

    if flag:
        all_none1 = np.all(pd.isnull(awesome_result_top3)) and np.all(pd.isnull(sparse_result_top3))
        all_none2 = np.all(pd.isnull(pruned_awesome_result_top3)) and np.all(pd.isnull(pruned_sparse_result_top3))

        # top NUM_CANDIDATES candidates selected, no pruning
        if not all_none1:
            np.testing.assert_array_almost_equal(awesome_result_top3, sparse_result_top3)
        else:
            assert len(awesome_result_top3) == len(sparse_result_top3)
        # top NUM_CANDIDATES candidates selected, below PRUNE_THRESHOLD similarity pruned
        if not all_none2:
            np.testing.assert_array_almost_equal(pruned_awesome_result_top3, pruned_sparse_result_top3)
        else:
            assert len(pruned_awesome_result_top3) == len(pruned_sparse_result_top3)
    else:
        assert awesome_result_top3 == sparse_result_top3
        assert pruned_awesome_result_top3 == pruned_sparse_result_top3


def test_awesome_cossim_topn_manually():
    # a simple case
    a_dense = [[0.2, 0.1, 0.0, 0.9, 0.3],
               [0.7, 0.0, 0.0, 0.2, 0.2],
               [0.0, 0.0, 0.0, 0.2, 0.1],
               [0.5, 0.4, 0.5, 0.0, 0.0]]

    b_dense = [[0.4, 0.2, 0.3, 0.2, 0.7],
               [0.9, 0.4, 0.5, 0.1, 0.4],
               [0.3, 0.8, 0.0, 0.2, 0.5],
               [0.3, 0.0, 0.1, 0.1, 0.6],
               [0.6, 0.1, 0.2, 0.8, 0.1],
               [0.9, 0.1, 0.6, 0.4, 0.3]]
    helper_awesome_cossim_topn_dense(a_dense, b_dense)
    helper_awesome_cossim_topn_dense(a_dense, b_dense, mem_manager_is_C=MEM_MANAGER_IS_C)
    for process in range(MAX_N_PROCESSES):
        n_jobs = process + 1
        helper_awesome_cossim_topn_dense(a_dense, b_dense, use_threads=USE_THREADS, n_jobs=n_jobs)
        helper_awesome_cossim_topn_dense(
            a_dense,
            b_dense,
            mem_manager_is_C=MEM_MANAGER_IS_C,
            use_threads=USE_THREADS,
            n_jobs=n_jobs
        )

    # boundary checking, there is no matching at all in this case
    c_dense = [[0.2, 0.1, 0.3, 0, 0],
               [0.7, 0.2, 0.7, 0, 0],
               [0.3, 0.9, 0.6, 0, 0],
               [0.5, 0.4, 0.5, 0, 0]]
    d_dense = [[0, 0, 0, 0.6, 0.9],
               [0, 0, 0, 0.1, 0.1],
               [0, 0, 0, 0.2, 0.6],
               [0, 0, 0, 0.8, 0.4],
               [0, 0, 0, 0.1, 0.3],
               [0, 0, 0, 0.7, 0.5]]
    helper_awesome_cossim_topn_dense(c_dense, d_dense)
    helper_awesome_cossim_topn_dense(c_dense, d_dense, mem_manager_is_C=MEM_MANAGER_IS_C)
    for process in range(MAX_N_PROCESSES):
        n_jobs = process + 1
        helper_awesome_cossim_topn_dense(c_dense, d_dense, use_threads=USE_THREADS, n_jobs=n_jobs)
        helper_awesome_cossim_topn_dense(
            c_dense,
            d_dense,
            mem_manager_is_C=MEM_MANAGER_IS_C,
            use_threads=USE_THREADS,
            n_jobs=n_jobs
        )


@pytest.mark.filterwarnings("ignore:Comparing a sparse matrix with a scalar greater than zero")
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive")
def test_awesome_cossim_top_one_zeros():
    # test with one row matrix with all zeros
    # helper_awesome_cossim_top_sparse uses a local function awesome_cossim_top
    nr_vocab = 1000
    density = 0.1
    for _ in range(3):
        a_sparse = csr_matrix(np.zeros((1, nr_vocab)))
        b_sparse = rand(800, nr_vocab, density=density, format='csr')
        helper_awesome_cossim_topn_sparse(a_sparse, b_sparse)
        helper_awesome_cossim_topn_sparse(a_sparse, b_sparse, mem_manager_is_C=MEM_MANAGER_IS_C)
        for process in range(MAX_N_PROCESSES):
            n_jobs = process + 1
            helper_awesome_cossim_topn_sparse(a_sparse, b_sparse, use_threads=USE_THREADS, n_jobs=n_jobs)
            helper_awesome_cossim_topn_sparse(
                a_sparse,
                b_sparse,
                mem_manager_is_C=MEM_MANAGER_IS_C,
                use_threads=USE_THREADS,
                n_jobs=n_jobs
            )


@pytest.mark.filterwarnings("ignore:Comparing a sparse matrix with a scalar greater than zero")
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive")
def test_awesome_cossim_top_all_zeros():
    # test with all zeros matrix
    # helper_awesome_cossim_top_sparse uses a local function awesome_cossim_top
    nr_vocab = 1000
    density = 0.1
    for _ in range(3):
        a_sparse = csr_matrix(np.zeros((2, nr_vocab)))
        b_sparse = rand(800, nr_vocab, density=density, format='csr')
        helper_awesome_cossim_topn_sparse(a_sparse, b_sparse)
        helper_awesome_cossim_topn_sparse(a_sparse, b_sparse, mem_manager_is_C=MEM_MANAGER_IS_C)
        for process in range(MAX_N_PROCESSES):
            n_jobs = process + 1
            helper_awesome_cossim_topn_sparse(a_sparse, b_sparse, use_threads=USE_THREADS, n_jobs=n_jobs)
            helper_awesome_cossim_topn_sparse(
                a_sparse,
                b_sparse,
                mem_manager_is_C=MEM_MANAGER_IS_C,
                use_threads=USE_THREADS,
                n_jobs=n_jobs
            )


@pytest.mark.filterwarnings("ignore:Comparing a sparse matrix with a scalar greater than zero")
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive")
def test_awesome_cossim_top_small_matrix():
    # test with small matrix
    nr_vocab = 1000
    density = 0.1
    for _ in range(10):
        a_sparse = rand(300, nr_vocab, density=density, format='csr')
        b_sparse = rand(800, nr_vocab, density=density, format='csr')
        helper_awesome_cossim_topn_sparse(a_sparse, b_sparse, False)
        helper_awesome_cossim_topn_sparse(a_sparse, b_sparse, False, mem_manager_is_C=MEM_MANAGER_IS_C)
        for process in range(MAX_N_PROCESSES):
            n_jobs = process + 1
            helper_awesome_cossim_topn_sparse(a_sparse, b_sparse, False, use_threads=USE_THREADS, n_jobs=n_jobs)
            helper_awesome_cossim_topn_sparse(
                a_sparse,
                b_sparse,
                False,
                mem_manager_is_C=MEM_MANAGER_IS_C,
                use_threads=USE_THREADS,
                n_jobs=n_jobs
            )


@pytest.mark.filterwarnings("ignore:Comparing a sparse matrix with a scalar greater than zero")
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive")
def test_awesome_cossim_top_large_matrix():
    # MB: I reduced the size of the matrix so the test also runs in small memory.
    # test with large matrix
    nr_vocab = 2 << 24
    density = 1e-6
    n_samples = 10000
    nnz = int(n_samples * nr_vocab * density)

    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(43)

    for _ in range(1):
        # scipy.sparse.rand has very high memory usage
        # see for details: https://github.com/scipy/scipy/issues/9699
        # a_sparse = rand(500, nr_vocab, density=density, format='csr')
        # b_sparse = rand(80000, nr_vocab, density=density, format='csr')

        # switching to alternative random method below, which is also a lot faster
        row = rng1.randint(500, size=nnz)
        cols = rng2.randint(nr_vocab, size=nnz)
        data = rng1.rand(nnz)

        a_sparse = coo_matrix((data, (row, cols)), shape=(n_samples, nr_vocab))
        a_sparse = a_sparse.tocsr()

        row = rng1.randint(n_samples, size=nnz)
        cols = rng2.randint(nr_vocab, size=nnz)
        data = rng1.rand(nnz)

        b_sparse = coo_matrix((data, (row, cols)), shape=(n_samples, nr_vocab))
        b_sparse = b_sparse.tocsr()

        helper_awesome_cossim_topn_sparse(a_sparse, b_sparse, False)
        helper_awesome_cossim_topn_sparse(a_sparse, b_sparse, False, mem_manager_is_C=MEM_MANAGER_IS_C)
        for process in range(MAX_N_PROCESSES):
            n_jobs = process + 1
            helper_awesome_cossim_topn_sparse(a_sparse, b_sparse, False, use_threads=USE_THREADS, n_jobs=n_jobs)
            helper_awesome_cossim_topn_sparse(
                a_sparse,
                b_sparse,
                False,
                mem_manager_is_C=MEM_MANAGER_IS_C,
                use_threads=USE_THREADS,
                n_jobs=n_jobs
            )
