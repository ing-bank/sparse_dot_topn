/* Copyright (c) 2023 ING Analytics Wholesale Banking
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <vector>

#include <sparse_dot_topn/common.hpp>

namespace sdtn::core {

template <typename idxT, iffInt<idxT> = true>
inline idxT sp_matmul_size(
    const idxT nrows,
    const idxT ncols,
    const idxT* __restrict A_indptr,
    const idxT* __restrict A_indices,
    const idxT* __restrict B_indptr,
    const idxT* __restrict B_indices,
    idxT* __restrict C_indptr
) {
    idxT nnz = 0;
    C_indptr[0] = 0;
    std::vector<idxT> mask(ncols, -1);
    for (idxT i = 0; i < nrows; i++) {
        idxT row_nnz = 0;
        idxT A_cidx_start = A_indptr[i];
        idxT A_cidx_end = A_indptr[i + 1];
        for (idxT A_cidx = A_cidx_start; A_cidx < A_cidx_end; ++A_cidx) {
            idxT j = A_indices[A_cidx];
            for (idxT kk = B_indptr[j]; kk < B_indptr[j + 1]; ++kk) {
                idxT k = B_indices[kk];
                if (mask[k] != i) {
                    mask[k] = i;
                    row_nnz++;
                }
            }
        }
        nnz += row_nnz;
        C_indptr[i + 1] = nnz;
    }
    return nnz;
}

/*
 * \brief Compute A.dot(B).
 *
 * \details This function will return a matrix C in CSR format, where C = A * B.
 * Note that `A` and `B` must be `CSR` format where the nonzero
 * elements of the `i`th row are located in ``data[indptr[i]:indptr[i+1]]``.
 * The column indices for row `i` are stored in
 * ``indices[indptr[i]:indptr[i+1]]``.
 *
 *  Copyright Scipy:
 *  This function is a modified version of `csr_binop_csr_general`
 *  Source: scipy/sparse/sparsetools/csr.h#L692
 *  License: BSD 3 https://github.com/scipy/scipy/blob/main/LICENSE.txt
 *  All modifications copyright INGA WB.
 *
 * \tparam eT   element type of the matrices
 * \tparam idxT integer type of the index arrays, must be at least 32 bit int
 * \param[in] nrows the number of rows in A
 * \param[in] ncols the number of columns in B
 * \param[in] A_data the nonzero elements of A
 * \param[in] A_indptr array containing the row indices for `A_data`
 * \param[in] A_indices array containing the column indices
 * \param[in] B_data the nonzero elements of B
 * \param[in] B_indptr array containing the row indices for `B_data`
 * \param[in] B_indices array containing the column indices
 * \param[out] C_data the nonzero elements of C
 * \param[out] C_indptr array containing the row indices for `C_data`
 * \param[out] C_indices array containing the column indices
 */
template <typename eT, typename idxT, iffInt<idxT> = true>
void sp_matmul(
    const idxT nrows,
    const idxT ncols,
    const eT* __restrict A_data,
    const idxT* __restrict A_indptr,
    const idxT* __restrict A_indices,
    const eT* __restrict B_data,
    const idxT* __restrict B_indptr,
    const idxT* __restrict B_indices,
    eT* __restrict C_data,
    idxT* __restrict C_indices
) {
    std::vector<idxT> next(ncols, -1);
    std::vector<eT> sums(ncols, 0);

    idxT nnz = 0;

    for (idxT i = 0; i < nrows; i++) {
        idxT head = -2;
        idxT length = 0;

        idxT jj_start = A_indptr[i];
        idxT jj_end = A_indptr[i + 1];
        for (idxT jj = jj_start; jj < jj_end; jj++) {
            idxT j = A_indices[jj];
            eT v = A_data[jj];

            idxT kk_start = B_indptr[j];
            idxT kk_end = B_indptr[j + 1];
            for (idxT kk = kk_start; kk < kk_end; kk++) {
                idxT k = B_indices[kk];

                sums[k] += v * B_data[kk];

                if (next[k] == -1) {
                    next[k] = head;
                    head = k;
                    length++;
                }
            }
        }

        for (idxT jj = 0; jj < length; jj++) {
            if (sums[head] != 0) {
                C_indices[nnz] = head;
                C_data[nnz] = sums[head];
                nnz++;
            }

            idxT temp = head;
            head = next[head];

            next[temp] = -1;  // clear arrays
            sums[temp] = 0;
        }
    }
}

#if defined(SDTN_OMP_ENABLED)
template <typename idxT, iffInt<idxT> = true>
inline idxT sp_matmul_size_mt(
    const idxT nrows,
    const idxT ncols,
    const idxT* __restrict A_indptr,
    const idxT* __restrict A_indices,
    const idxT* __restrict B_indptr,
    const idxT* __restrict B_indices,
    idxT* __restrict C_indptr
) {
    idxT nnz = 0;
    C_indptr[0] = 0;
#pragma omp parallel default(none) shared(                                    \
        nrows, ncols, A_indptr, A_indices, B_indptr, B_indices, C_indptr, nnz \
)
    {
        std::vector<idxT> mask(ncols, -1);
#pragma omp for reduction(+ : nnz)
        for (idxT i = 0; i < nrows; i++) {
            idxT row_nnz = 0;
            idxT A_cidx_start = A_indptr[i];
            idxT A_cidx_end = A_indptr[i + 1];
            for (idxT A_cidx = A_cidx_start; A_cidx < A_cidx_end; ++A_cidx) {
                idxT j = A_indices[A_cidx];
                for (idxT kk = B_indptr[j]; kk < B_indptr[j + 1]; ++kk) {
                    idxT k = B_indices[kk];
                    if (mask[k] != i) {
                        mask[k] = i;
                        row_nnz++;
                    }
                }
            }
            C_indptr[i + 1] = row_nnz;
            nnz += row_nnz;
        }
    }  // pragma omp parallel
    for (idxT i = 2; i < nrows + 1; i++) {
        C_indptr[i] += C_indptr[i - 1];
    }
    return nnz;
}
/*
 * \brief Compute A.dot(B).
 *
 * \details This function will return a matrix C in CSR format, where C = A
 * * B. Note that `A` and `B` must be `CSR` format where the nonzero
 * elements of the `i`th row are located in ``data[indptr[i]:indptr[i+1]]``.
 * The column indices for row `i` are stored in
 * ``indices[indptr[i]:indptr[i+1]]``.
 *
 *  Copyright Scipy:
 *  This function is a modified version of `csr_binop_csr_general`
 *  Source: scipy/sparse/sparsetools/csr.h#L692
 *  License: BSD 3 https://github.com/scipy/scipy/blob/main/LICENSE.txt
 *  All modifications copyright INGA WB.
 *
 * \tparam eT   element type of the matrices
 * \tparam idxT integer type of the index arrays, must be at least 32 bit
 * int \param[in] nrows the number of rows in A \param[in] ncols the number
 * of columns in B \param[in] A_data the nonzero elements of A \param[in]
 * A_indptr array containing the row indices for `A_data` \param[in]
 * A_indices array containing the column indices \param[in] B_data the
 * nonzero elements of B \param[in] B_indptr array containing the row
 * indices for `B_data` \param[in] B_indices array containing the column
 * indices \param[out] C_data the nonzero elements of C \param[out] C_indptr
 * array containing the row indices for `C_data` \param[out] C_indices array
 * containing the column indices
 */
template <typename eT, typename idxT, iffInt<idxT> = true>
void sp_matmul_mt(
    const idxT nrows,
    const idxT ncols,
    const int n_threads,
    const eT* __restrict A_data,
    const idxT* __restrict A_indptr,
    const idxT* __restrict A_indices,
    const eT* __restrict B_data,
    const idxT* __restrict B_indptr,
    const idxT* __restrict B_indices,
    eT* __restrict C_data,
    idxT* __restrict C_indptr,
    idxT* __restrict C_indices
) {
#pragma omp parallel num_threads(n_threads) default(none) \
    shared(nrows,                                         \
               ncols,                                     \
               A_data,                                    \
               A_indptr,                                  \
               A_indices,                                 \
               B_data,                                    \
               B_indptr,                                  \
               B_indices,                                 \
               C_data,                                    \
               C_indptr,                                  \
               C_indices)
    {
        std::vector<idxT> next(ncols, -1);
        std::vector<eT> sums(ncols, 0);

#pragma omp for
        for (idxT i = 0; i < nrows; i++) {
            idxT nnz = 0;
            idxT head = -2;
            idxT length = 0;
            idxT* local_C_indices = C_indices + C_indptr[i];
            eT* local_C_data = C_data + C_indptr[i];

            idxT jj_start = A_indptr[i];
            idxT jj_end = A_indptr[i + 1];
            for (idxT jj = jj_start; jj < jj_end; jj++) {
                idxT j = A_indices[jj];
                eT v = A_data[jj];

                idxT kk_start = B_indptr[j];
                idxT kk_end = B_indptr[j + 1];
                for (idxT kk = kk_start; kk < kk_end; kk++) {
                    idxT k = B_indices[kk];

                    sums[k] += v * B_data[kk];

                    if (next[k] == -1) {
                        next[k] = head;
                        head = k;
                        length++;
                    }
                }
            }

            for (idxT jj = 0; jj < length; jj++) {
                if (sums[head] != 0) {
                    local_C_indices[nnz] = head;
                    local_C_data[nnz] = sums[head];
                    nnz++;
                }

                idxT temp = head;
                head = next[head];

                next[temp] = -1;  // clear arrays
                sums[temp] = 0;
            }
        }
    }  // #pragma omp parallel
}
#endif  // SDTN_OMP_ENABLED
}  // namespace sdtn::core
