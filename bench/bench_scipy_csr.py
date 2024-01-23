# Copyright (c) 2023 ING Analytics Wholesale Banking
from __future__ import annotations

import re
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul, sp_matmul_topn

# A.shape: 331500, 193190
# B.shape: 331500, 193190

# Max number of rows possible is 331500
N_ROWS = 20_000
INPUT_FILE = "sec__edgar_company_info.csv.gz"

dir_path = Path(__file__).parent.resolve()
path_A = dir_path / f"tfidf_A_{N_ROWS}.npz"
path_B = dir_path / f"tfidf_B_{N_ROWS}.npz"

if not (path_A.exists() and path_B.exists()):

    def ngrams(string, n=3):
        string = re.sub(r"[,-./]|\sBD", r"", string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return ["".join(ngram) for ngram in ngrams]

    dir_path = Path(__file__).parent.resolve()
    names = pd.read_csv(dir_path / INPUT_FILE)
    company_names = names["Company Name"].str.lower().sample(frac=1).reset_index(drop=True)
    del names

    hn = company_names.size // 2

    vectorizer = TfidfVectorizer(min_df=1).fit(company_names)
    A = vectorizer.transform(company_names[:hn])
    B = vectorizer.transform(company_names[hn:])

    rng = np.random.default_rng()

    A_srows = np.sort(rng.choice(np.arange(hn), size=N_ROWS, replace=False))
    B_srows = np.sort(rng.choice(np.arange(hn), size=N_ROWS, replace=False))

    save_npz(path_A, A[A_srows, :])
    save_npz(path_B, B[B_srows, :].transpose().tocsr())

A = load_npz(path_A)
B = load_npz(path_B)


def bench_scipy_csr():
    A.dot(B)


__benchmarks__ = [
    (bench_scipy_csr, partial(sp_matmul, A, B, n_threads=1), "Scipy vs sp_matmul                    | n_threads: 1"),
    (bench_scipy_csr, partial(sp_matmul, A, B, n_threads=2), "Scipy vs sp_matmul                    | n_threads: 2"),
    (bench_scipy_csr, partial(sp_matmul, A, B, n_threads=4), "Scipy vs sp_matmul                    | n_threads: 4"),
    (bench_scipy_csr, partial(sp_matmul, A, B, n_threads=8), "Scipy vs sp_matmul                    | n_threads: 8"),
    (bench_scipy_csr, partial(sp_matmul_topn, A, B, 10), "Scipy vs sp_matmul_topn | top_n: 10   | n_threads: 1"),
    (bench_scipy_csr, partial(sp_matmul_topn, A, B, 20), "Scipy vs sp_matmul_topn | top_n: 20   | n_threads: 1"),
    (bench_scipy_csr, partial(sp_matmul_topn, A, B, 30), "Scipy vs sp_matmul_topn | top_n: 30   | n_threads: 1"),
    (bench_scipy_csr, partial(sp_matmul_topn, A, B, 100), "Scipy vs sp_matmul_topn | top_n: 100  | n_threads: 1"),
    (bench_scipy_csr, partial(sp_matmul_topn, A, B, 1000), "Scipy vs sp_matmul_topn | top_n: 1000 | n_threads: 1"),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 10, n_threads=2),
        "Scipy vs sp_matmul_topn | top_n: 10   | n_threads: 2",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 20, n_threads=2),
        "Scipy vs sp_matmul_topn | top_n: 20   | n_threads: 2",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 30, n_threads=2),
        "Scipy vs sp_matmul_topn | top_n: 30   | n_threads: 2",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 100, n_threads=2),
        "Scipy vs sp_matmul_topn | top_n: 100  | n_threads: 2",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 1000, n_threads=2),
        "Scipy vs sp_matmul_topn | top_n: 1000 | n_threads: 2",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 10, n_threads=4),
        "Scipy vs sp_matmul_topn | top_n: 10   | n_threads: 4",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 20, n_threads=4),
        "Scipy vs sp_matmul_topn | top_n: 20   | n_threads: 4",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 30, n_threads=4),
        "Scipy vs sp_matmul_topn | top_n: 30   | n_threads: 4",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 100, n_threads=4),
        "Scipy vs sp_matmul_topn | top_n: 100  | n_threads: 4",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 1000, n_threads=4),
        "Scipy vs sp_matmul_topn | top_n: 1000 | n_threads: 4",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 10, n_threads=8),
        "Scipy vs sp_matmul_topn | top_n: 10   | n_threads: 8",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 20, n_threads=8),
        "Scipy vs sp_matmul_topn | top_n: 20   | n_threads: 8",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 30, n_threads=8),
        "Scipy vs sp_matmul_topn | top_n: 30   | n_threads: 8",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 100, n_threads=8),
        "Scipy vs sp_matmul_topn | top_n: 100  | n_threads: 8",
    ),
    (
        bench_scipy_csr,
        partial(sp_matmul_topn, A, B, 1000, n_threads=8),
        "Scipy vs sp_matmul_topn | top_n: 1000 | n_threads: 8",
    ),
]
