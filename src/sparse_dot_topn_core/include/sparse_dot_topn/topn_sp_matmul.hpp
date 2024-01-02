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
#include <vector>

namespace sdtn::core {

template <typename T>
using iffInt = std::enable_if_t<std::is_integral_v<T>, bool>;

template <typename T>
using iffFloat = std::enable_if_t<std::is_floating_point_v<T>, bool>;

template <typename eT, typename idxT>
struct Candidate {
    idxT index;
    eT value;

    bool operator<(const Candidate& a) const { return a.value < value; }
};

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
 *
 * \tparam eT   element type of the matrices
 * \tparam idxT integer type of the index arrays, must be at least 32 bit int
 * \param[in] top_n the top n values to store
 * \param[in] nrows the number of rows in A
 * \param[in] ncols the number of columns in B
 * \param[in] threshold minimum values required to store
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
inline void topn_sp_matmul(
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
    eT* __restrict C_data,
    idxT* __restrict C_indptr,
    idxT* __restrict C_indices
) {
    std::vector<idxT> next(ncols, -1);
    std::vector<eT> sums(ncols, 0);

    std::vector<Candidate<eT, idxT>> candidates;

    idxT nnz = 0;

    C_indptr[0] = 0;

    for (idxT i = 0; i < nrows; i++) {
        idxT head = -2;
        idxT length = 0;

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
            if (sums[head] > threshold) {
                // append the nonzero elements
                Candidate<eT, idxT> c;
                c.index = head;
                c.value = sums[head];
                candidates.push_back(c);
            }

            idxT temp = head;
            // iterate over columns
            head = next[head];

            // clear arrays
            next[temp] = -1;
            sums[temp] = 0;
        }

        auto len = static_cast<idxT>(candidates.size());
        if (len > top_n) {
            std::partial_sort(
                candidates.begin(), candidates.begin() + top_n, candidates.end()
            );
            len = top_n;
        } else {
            std::sort(candidates.begin(), candidates.end());
        }

        for (idxT a = 0; a < len; a++) {
            C_indices[nnz] = candidates[a].index;
            C_data[nnz] = candidates[a].value;
            nnz++;
        }
        candidates.clear();

        C_indptr[i + 1] = nnz;
    }
}

}  // namespace sdtn::core