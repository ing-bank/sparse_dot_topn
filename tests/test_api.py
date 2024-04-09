import sys
from itertools import product

import numpy as np
import pytest
from scipy import sparse
from sparse_dot_topn import _has_openmp_support, sp_matmul, sp_matmul_topn, zip_sp_matmul_topn

from ._resources import _assert_array_equal, _assert_smat_equal, _get_topn_elements


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_sp_matmul_default(rng, dtype):
    A = sparse.random(100, 100, density=0.1, format="csr", dtype=dtype, random_state=rng)
    B = sparse.random(100, 100, density=0.1, format="csr", dtype=dtype, random_state=rng)
    C = sp_matmul(A, B)
    C_ref = A.dot(B)
    _assert_smat_equal(C, C_ref)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_sp_matmul_nthreads(rng, dtype):
    A = sparse.random(100, 10, density=0.5, format="csr", dtype=dtype, random_state=rng)
    B = sparse.random(10, 100, density=0.5, format="csr", dtype=dtype, random_state=rng)
    C_ref = A.dot(B)
    if not _has_openmp_support:
        with pytest.warns(UserWarning):
            C = sp_matmul(A, B, n_threads=2)
    else:
        C = sp_matmul(A, B, n_threads=2)
    _assert_smat_equal(C, C_ref)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_sp_matmul_topn_default(rng, dtype):
    A = sparse.random(100, 100, density=0.1, format="csr", dtype=dtype, random_state=rng)
    B = sparse.random(100, 100, density=0.1, format="csr", dtype=dtype, random_state=rng)
    C = sp_matmul_topn(A, B, top_n=B.shape[1])
    C_ref = A.dot(B)
    _assert_smat_equal(C, C_ref)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_sp_matmul_topn_topn(rng, dtype):
    A = sparse.random(100, 10, density=0.5, format="csr", dtype=dtype, random_state=rng)
    B = sparse.random(10, 100, density=0.5, format="csr", dtype=dtype, random_state=rng)
    C_ref = A.dot(B)
    C_10 = sp_matmul_topn(A, B, top_n=10)
    C_30 = sp_matmul_topn(A, B, top_n=30)
    for i in range(A.shape[0]):
        _assert_array_equal(C_10[i, :].data, _get_topn_elements(C_ref[i, :].data, 10))
        _assert_array_equal(C_30[i, :].data, _get_topn_elements(C_ref[i, :].data, 30))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_sp_matmul_topn_nthreads(rng, dtype):
    A = sparse.random(100, 10, density=0.5, format="csr", dtype=dtype, random_state=rng)
    B = sparse.random(10, 100, density=0.5, format="csr", dtype=dtype, random_state=rng)
    C_ref = A.dot(B)
    if not _has_openmp_support:
        with pytest.warns(UserWarning):
            C = sp_matmul_topn(A, B, top_n=B.shape[1], n_threads=2)
    else:
        C = sp_matmul_topn(A, B, top_n=B.shape[1], n_threads=2)
    _assert_smat_equal(C, C_ref)

    C_10 = sp_matmul_topn(A, B, top_n=10, n_threads=2)
    C_30 = sp_matmul_topn(A, B, top_n=30, n_threads=2)
    for i in range(A.shape[0]):
        _assert_array_equal(C_10[i, :].data, _get_topn_elements(C_ref[i, :].data, 10))
        _assert_array_equal(C_30[i, :].data, _get_topn_elements(C_ref[i, :].data, 30))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_sp_matmul_topn_sorted(rng, dtype):
    A = sparse.random(100, 10, density=0.5, format="csr", dtype=dtype, random_state=rng)
    B = sparse.random(10, 100, density=0.5, format="csr", dtype=dtype, random_state=rng)
    C_ref = A.dot(B)

    C_10 = sp_matmul_topn(A, B, top_n=10, sort=True)
    C_30 = sp_matmul_topn(A, B, top_n=30, sort=True, n_threads=2)
    for i in range(A.shape[0]):
        sorted_row = np.sort(C_ref[i, :].data)[::-1]
        _assert_array_equal(C_10[i, :].data, sorted_row[:10])
        _assert_array_equal(C_30[i, :].data, sorted_row[:30])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_sp_matmul_topn_density(rng, dtype):
    A = sparse.random(200, 200, density=0.9, format="csr", dtype=dtype, random_state=rng)
    B = sparse.random(200, 200, density=0.9, format="csr", dtype=dtype, random_state=rng)
    C = sp_matmul_topn(A, B, top_n=B.shape[1], density=0.01)
    C_ref = A.dot(B)
    _assert_smat_equal(C, C_ref)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_sp_matmul_topn_threshold(rng, dtype):
    A = sparse.random(100, 100, density=0.1, format="csr", dtype=dtype, random_state=rng)
    B = sparse.random(100, 100, density=0.1, format="csr", dtype=dtype, random_state=rng)
    threshold = 0.0 if np.issubdtype(A.data.dtype, np.integer) else 0.5
    C = sp_matmul_topn(A, B, top_n=B.shape[1], threshold=threshold)
    C_ref = A.dot(B)
    assert C.data.min() > threshold
    assert C.data.size < C_ref.data.size


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_sp_matmul_topn_idx_dtype(rng, dtype):
    A = sparse.random(100, 100, density=0.1, format="csr", dtype=dtype, random_state=rng)
    B = sparse.random(100, 100, density=0.1, format="csr", dtype=dtype, random_state=rng)
    C = sp_matmul_topn(A, B, top_n=B.shape[1], idx_dtype=np.int64)
    C_ref = A.dot(B)
    _assert_smat_equal(C, C_ref)


_FORMATS = ["coo", "csr", "csc"]


@pytest.mark.parametrize("formats", product(_FORMATS, _FORMATS))
def test_sp_matmul_topn_formats(rng, formats):
    A = sparse.random(200, 100, density=0.1, format=formats[0], random_state=rng)
    B = sparse.random(100, 300, density=0.1, format=formats[1], random_state=rng)

    C = sp_matmul_topn(A, B, top_n=B.shape[1])
    C_ref = A.tocsr().dot(B.tocsr())
    _assert_smat_equal(C, C_ref)


def test_sp_matmul_topn_transpose(rng):
    A = sparse.random(200, 100, density=0.1, format="csr", random_state=rng)
    B = sparse.random(100, 300, density=0.1, format="csr", random_state=rng)

    C = sp_matmul_topn(A.transpose(), B.transpose(), top_n=B.shape[1])
    C_ref = A.dot(B)
    _assert_smat_equal(C, C_ref)


_KWARGS = [
    {
        "A": {"m": 10, "n": 10, "density": 0.1, "format": "csr"},
        "B": {"m": 10, "n": 10, "density": 0.1, "format": "csr", "T": False},
    },
    {
        "A": {"m": 100, "n": 100, "density": 0.1, "format": "csr"},
        "B": {"m": 100, "n": 100, "density": 0.1, "format": "csr", "T": False},
    },
    {
        "A": {"m": 1000, "n": 1000, "density": 0.05, "format": "csr"},
        "B": {"m": 1000, "n": 1000, "density": 0.05, "format": "csr", "T": False},
    },
    {
        "A": {"m": 2000, "n": 2000, "density": 0.01, "format": "csr"},
        "B": {"m": 2000, "n": 2000, "density": 0.01, "format": "csr", "T": False},
    },
    {
        "A": {"m": 10000, "n": 10000, "density": 0.001, "format": "csr"},
        "B": {"m": 10000, "n": 10000, "density": 0.001, "format": "csr", "T": False},
    },
    {
        "A": {"m": 1000, "n": 100, "density": 0.1, "format": "csr"},
        "B": {"m": 100, "n": 100, "density": 0.1, "format": "csr", "T": False},
    },
    {
        "A": {"m": 1000, "n": 100, "density": 0.1, "format": "csr"},
        "B": {"m": 100, "n": 20, "density": 0.1, "format": "csr", "T": False},
    },
    {
        "A": {"m": 1000, "n": 100, "density": 0.1, "format": "csr"},
        "B": {"m": 200, "n": 100, "density": 0.1, "format": "csr", "T": True},
    },
    {
        "A": {"m": 1000, "n": 100, "density": 0.1, "format": "csr"},
        "B": {"m": 200, "n": 100, "density": 0.1, "format": "csc", "T": True},
    },
]


@pytest.mark.parametrize("kwargs", _KWARGS)
def test_sp_matmul_topn_broadcasting(rng, kwargs):
    A = sparse.random(**kwargs["A"], random_state=rng)
    T_b = kwargs["B"].pop("T")
    B = sparse.random(**kwargs["B"], random_state=rng)

    topn = B.shape[0] if T_b else B.shape[1]
    C_ref = A.dot(B.T) if T_b else A.dot(B)
    C = sp_matmul_topn(A, B, top_n=topn)
    _assert_smat_equal(C, C_ref)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_zip_sp_matmul_topn(rng, dtype):
    # matching 100 names against 600 gt-names, where gt has been split into three parts
    A = sparse.random(100, 2000, density=0.1, format="csr", dtype=dtype, random_state=rng)
    B = sparse.random(600, 2000, density=0.1, format="csr", dtype=dtype, random_state=rng)

    # reference
    C_ref = sp_matmul_topn(A, B.T, top_n=10, threshold=0.01, sort=True)

    # zipped C-matrix
    Bs = [B[:100], B[100:300], B[300:]]
    Cs = [sp_matmul_topn(A, Bi.T, top_n=10, threshold=0.01, sort=True) for Bi in Bs]
    C_zip = zip_sp_matmul_topn(top_n=10, C_mats=Cs)

    # comparison of index-pointers, data, indices
    _assert_array_equal(C_zip.indptr, C_ref.indptr)
    _assert_array_equal(C_zip.data, C_ref.data)
    # NB indices are not necessarily equal. For example in case of multiple equal scores close to the topn boundary,
    # sp_matmul_topn and the zipped approach may pick different matches. In such cases for sp_matmul_topn
    # the insertion order in the maxheap is leading, which is impossible to replicate in zip_sp_matmul_topn,
    # as the B matrices are have been split and get inserted in separate maxheap objects.
    _assert_array_equal(C_zip.indices, C_ref.indices)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="not all dtypes supported in scipy vstack for python 3.8")
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_stack_zip_sp_matmul_topn(rng, dtype):
    # matching 1000 names against 600 gt-names,
    # where gt has been split into three parts, and names-to-match in five parts
    A = sparse.random(1000, 2000, density=0.1, format="csr", dtype=dtype, random_state=rng)
    B = sparse.random(600, 2000, density=0.1, format="csr", dtype=dtype, random_state=rng)

    # reference
    C_ref = sp_matmul_topn(A, B.T, top_n=10, threshold=0.01, sort=True)

    # zipped C-matrix
    As = [A[i * 200 : (i + 1) * 200] for i in range(5)]  # names-to-match split
    Bs = [B[:100], B[100:300], B[300:]]  # GT split

    Cs = [[sp_matmul_topn(Aj, Bi.T, top_n=10, threshold=0.01, sort=True) for Bi in Bs] for Aj in As]
    C_zips = [zip_sp_matmul_topn(top_n=10, C_mats=Cis) for Cis in Cs]

    # stack over names-to-match subparts
    C_stack = sparse.vstack(C_zips, dtype=dtype)

    # comparison of index-pointers, data, indices
    _assert_array_equal(C_stack.indptr, C_ref.indptr)
    _assert_array_equal(C_stack.data, C_ref.data)
    # NB indices are not necessarily equal. For example in case of multiple equal scores close to the topn boundary,
    # sp_matmul_topn and the zipped approach may pick different matches. In such cases for sp_matmul_topn
    # the insertion order in the maxheap is leading, which is impossible to replicate in zip_sp_matmul_topn,
    # as the B matrices are have been split and get inserted in separate maxheap objects.
    _assert_array_equal(C_stack.indices, C_ref.indices)
