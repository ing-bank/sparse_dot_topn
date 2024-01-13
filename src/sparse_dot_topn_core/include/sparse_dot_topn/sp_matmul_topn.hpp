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

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#include <sparse_dot_topn/common.hpp>
#include <sparse_dot_topn/maxheap.hpp>

namespace sdtn::core {

template <typename idxT, iffInt<idxT> = true>
inline idxT sp_matmul_topn_size(
    const idxT top_n,
    const idxT nrows,
    const idxT ncols,
    const idxT* __restrict A_indptr,
    const idxT* __restrict A_indices,
    const idxT* __restrict B_indptr,
    const idxT* __restrict B_indices
) {
    idxT nnz = 0;
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
        nnz += std::min(top_n, row_nnz);
    }
    return nnz;
}

/**
 * \brief Compute A.dot(B) keeping only the top n results.
 *
 * \details This function will return a matrix C in CSR format, where
 * C = [sorted top n results > lower_bound for each row of A * B].
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
 * \param[in] top_n the top n values to store
 * \param[in] nrows the number of rows in A
 * \param[in] ncols the number of columns in B
 * \param[in] threshold minimum values required to store
 * \param[in] density the expected density of nonzero elements C
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
template <typename eT, typename idxT, bool insertion_sort, iffInt<idxT> = true>
inline void sp_matmul_topn(
    const idxT top_n,
    const idxT nrows,
    const idxT ncols,
    const eT threshold,
    const eT* __restrict A_data,
    const idxT* __restrict A_indptr,
    const idxT* __restrict A_indices,
    const eT* __restrict B_data,
    const idxT* __restrict B_indptr,
    const idxT* __restrict B_indices,
    std::vector<eT>& C_data,
    std::vector<idxT>& C_indptr,
    std::vector<idxT>& C_indices
) {
    std::vector<idxT> next(ncols, -1);
    std::vector<eT> sums(ncols, 0);

    auto max_heap = MaxHeap<eT, idxT>(top_n, threshold);
    idxT nnz = 0;

    C_indptr[0] = 0;

    for (idxT i = 0; i < nrows; i++) {
        idxT head = -2;
        idxT length = 0;
        eT min = max_heap.reset();

        // A_cidx: column index for A
        idxT A_cidx_start = A_indptr[i];
        idxT A_cidx_end = A_indptr[i + 1];
        for (idxT A_cidx = A_cidx_start; A_cidx < A_cidx_end; A_cidx++) {
            idxT j = A_indices[A_cidx];
            // value of A in (i,j)
            eT v = A_data[A_cidx];

            idxT B_ridx_start = B_indptr[j];
            idxT B_ridx_end = B_indptr[j + 1];
            for (idxT B_ridx = B_ridx_start; B_ridx < B_ridx_end; B_ridx++) {
                idxT k = B_indices[B_ridx];  // kth column of B in row j

                // multiply with value of B in (j,k) and accumulate to the
                // result for kth column of row i
                sums[k] += v * B_data[B_ridx];

                if (next[k] == -1) {
                    // keep a linked list, every element points to the next
                    // column index
                    next[k] = head;
                    head = k;
                    length++;
                }
            }
        }

        for (idxT jj = 0; jj < length; jj++) {
            // length = number of columns set (may include 0s)
            if (sums[head] > min) {
                min = max_heap.push_pop(head, sums[head]);
            }

            idxT temp = head;
            // iterate over columns
            head = next[head];

            // clear arrays
            next[temp] = -1;
            sums[temp] = 0;
        }

        if constexpr (insertion_sort) {
            // sort the heap s.t. the original matrix order is maintained
            max_heap.insertion_sort();
        } else {
            // sort the heap s.t. the first value is the largest
            max_heap.value_sort();
        }
        int n_set = max_heap.get_n_set();
        for (int ii = 0; ii < n_set; ++ii) {
            C_indices.push_back(max_heap.heap[ii].idx);
            C_data.push_back(max_heap.heap[ii].val);
        }
        nnz += n_set;
        C_indptr[i + 1] = nnz;
    }
}

#if defined(SDTN_OMP_ENABLED)
template <typename idxT, iffInt<idxT> = true>
inline idxT sp_matmul_topn_size_mt(
    const idxT top_n,
    const idxT nrows,
    const idxT* __restrict A_indptr,
    const idxT* __restrict A_indices,
    const idxT* __restrict B_indptr
) {
    idxT nnz = 0;
#pragma omp parallel for default(none) \
    shared(top_n, A_indptr, A_indices, B_indptr) reduction(+ : nnz)
    for (idxT i = 0; i < nrows; i++) {
        idxT row_nnz = 0;
        idxT A_cidx_start = A_indptr[i];
        idxT A_cidx_end = A_indptr[i + 1];
        for (idxT A_cidx = A_cidx_start; A_cidx < A_cidx_end; ++A_cidx) {
            idxT j = A_indices[A_cidx];
            row_nnz += (B_indptr[j + 1] - B_indptr[j]);
        }
        nnz += std::min(top_n, row_nnz);
    }
    return nnz;
}

template <typename idxT, iffInt<idxT> = true>
inline idxT sp_matmul_topn_size_mt(
    const idxT top_n,
    const idxT nrows,
    const idxT ncols,
    const idxT* __restrict A_indptr,
    const idxT* __restrict A_indices,
    const idxT* __restrict B_indptr,
    const idxT* __restrict B_indices
) {
    idxT nnz = 0;
#pragma omp parallel default(none) \
    shared(top_n, nrows, ncols, A_indptr, A_indices, B_indptr)
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
            nnz += std::min(top_n, row_nnz);
        }
    }
    return nnz;
}

/**
 * \brief Compute A.dot(B) keeping only the top n results.
 *
 * \details This function will return a matrix C in CSR format, where
 * C = [sorted top n results > lower_bound for each row of A * B].
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
 * \param[in] top_n the top n values to store
 * \param[in] nrows the number of rows in A
 * \param[in] ncols the number of columns in B
 * \param[in] threshold minimum values required to store
 * \param[in] n_threads number of threads to use
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
template <typename eT, typename idxT, bool insertion_sort, iffInt<idxT> = true>
inline std::tuple<size_t, eT*, idxT*, idxT*> sp_matmul_topn_mt(
    const idxT top_n,
    const idxT nrows,
    const idxT ncols,
    const eT threshold,
    const int n_threads,
    const eT* __restrict A_data,
    const idxT* __restrict A_indptr,
    const idxT* __restrict A_indices,
    const eT* __restrict B_data,
    const idxT* __restrict B_indptr,
    const idxT* __restrict B_indices
) {
    auto values = std::unique_ptr<eT[]>(new eT[nrows * top_n]);
    auto indices = std::unique_ptr<idxT[]>(new idxT[nrows * top_n]);
    auto row_nset = std::unique_ptr<idxT[]>(new idxT[nrows]);
#pragma omp parallel num_threads(n_threads) \
    shared(top_n,                           \
               nrows,                       \
               ncols,                       \
               threshold,                   \
               A_data,                      \
               A_indptr,                    \
               A_indices,                   \
               B_data,                      \
               B_indptr,                    \
               B_indices,                   \
               values,                      \
               indices,                     \
               row_nset)
    {
        std::vector<idxT> next(ncols, -1);
        std::vector<eT> sums(ncols, 0);

        auto max_heap = MaxHeap<eT, idxT>(top_n, threshold);

#pragma omp for
        for (idxT i = 0; i < nrows; i++) {
            idxT head = -2;
            idxT length = 0;

            idxT offset = i * top_n;
            eT* local_vals = values.get() + offset;
            idxT* local_idxs = indices.get() + offset;

            eT min = max_heap.reset();

            // A_cidx: column index for A
            idxT A_cidx_start = A_indptr[i];
            idxT A_cidx_end = A_indptr[i + 1];
            for (idxT A_cidx = A_cidx_start; A_cidx < A_cidx_end; A_cidx++) {
                idxT j = A_indices[A_cidx];
                // value of A in (i,j)
                eT v = A_data[A_cidx];

                idxT B_ridx_start = B_indptr[j];
                idxT B_ridx_end = B_indptr[j + 1];
                for (idxT B_ridx = B_ridx_start; B_ridx < B_ridx_end;
                     B_ridx++) {
                    idxT k = B_indices[B_ridx];  // kth column of B in row j

                    // multiply with value of B in (j,k) and accumulate to the
                    // result for kth column of row i
                    sums[k] += v * B_data[B_ridx];

                    if (next[k] == -1) {
                        // keep a linked list, every element points to the next
                        // column index
                        next[k] = head;
                        head = k;
                        length++;
                    }
                }
            }

            for (idxT jj = 0; jj < length; jj++) {
                // length = number of columns set (may include 0s)
                if (sums[head] > min) {
                    min = max_heap.push_pop(head, sums[head]);
                }

                idxT temp = head;
                // iterate over columns
                head = next[head];

                // clear arrays
                next[temp] = -1;
                sums[temp] = 0;
            }

            if constexpr (insertion_sort) {
                // sort the heap s.t. the original matrix order is maintained
                max_heap.insertion_sort();
            } else {
                // sort the heap s.t. the first value is the largest
                max_heap.value_sort();
            }
            int n_set = max_heap.get_n_set();
            for (int ii = 0; ii < n_set; ++ii) {
                local_idxs[ii] = max_heap.heap[ii].idx;
                local_vals[ii] = max_heap.heap[ii].val;
            }
            row_nset[i] = n_set;
        }
    }  // #pragma omp parallel

    // check how many non-zero elements are in C
    size_t total_nonzero
        = std::accumulate(row_nset.get(), row_nset.get() + nrows, 0);
    idxT* C_indptr = new idxT[nrows + 1];
    C_indptr[0] = 0;
    idxT* C_indices = new idxT[total_nonzero];
    eT* C_data = new eT[total_nonzero];
    // create ptr that will be shifted
    idxT* C_idx_ptr = C_indices;
    eT* C_data_ptr = C_data;

    int nnz = 0;
    idxT* idx_ptr = indices.get();
    eT* vals_ptr = values.get();

    for (int i = 0; i < nrows; ++i) {
        idxT n_set = row_nset[i];
        std::memcpy(C_idx_ptr, idx_ptr, n_set * sizeof(idxT));
        std::memcpy(C_data_ptr, vals_ptr, n_set * sizeof(eT));
        nnz += n_set;
        C_indptr[i + 1] = nnz;
        C_idx_ptr += n_set;
        C_data_ptr += n_set;
        idx_ptr += top_n;
        vals_ptr += top_n;
    }
    return std::make_tuple(total_nonzero, C_data, C_indices, C_indptr);
}  // sp_matmul_topn_mt
#endif  // SDTN_OMP_ENABLED

}  // namespace sdtn::core
