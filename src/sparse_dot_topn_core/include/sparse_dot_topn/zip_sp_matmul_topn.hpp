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

#include <limits>
#include <vector>

#include <sparse_dot_topn/common.hpp>
#include <sparse_dot_topn/maxheap.hpp>

namespace sdtn::core {

/**
 * \brief Zip and compute Z = zip_j C_j = zip_j A.dot(B_j) keeping only the
 * top-n of the zipped results.
 *
 * \details This function will return a zipped matrix Z in CSR format, zip_j
 * C_j, where Z = [sorted top n results > lower_bound for each row of C_j],
 * where C_j = A.dot(B_j) and where B has been split row-wise into sub-matrices
 * B_j. Note that `C_j` must be `CSR` format where the nonzero elements of the
 * `i`th row are located in ``data[indptr[i]:indptr[i+1]]``. The column indices
 * for row `i` are stored in
 * ``indices[indptr[i]:indptr[i+1]]``.
 *
 * \tparam eT   element type of the matrices
 * \tparam idxT integer type of the index arrays, must be at least 32 bit int
 * \param[in] top_n the top n values to store
 * \param[in] nrowsA the number of rows in A
 * \param[in] ncolsB_vec the number of columns in each B_i sub-matrix
 * \param[in] C_data_vec vector of the nonzero elements of each C_data_j
 *     sub-matrix
 * \param[in] C_indptr_vec vector of arrays containing the row indices for
 *     `C_data_j` sub-matrices
 * \param[in] C_indices_vec vector of arrays containing the column indices
       for the C_j sub-matrices
 * \param[out] Z_data the nonzero elements of zipped Z matrix
 * \param[out] Z_indptr array containing the row indices for zipped `Z_data`
 * \param[out] Z_indices array containing the zipped column indices
 */
template <typename eT, typename idxT, iffInt<idxT> = true>
inline void zip_sp_matmul_topn(
    const idxT top_n,
    const idxT nrows,
    const idxT* B_ncols,
    const std::vector<const eT*>& C_data,
    const std::vector<const idxT*>& C_indptrs,
    const std::vector<const idxT*>& C_indices,
    eT* __restrict Z_data,
    idxT* __restrict Z_indptr,
    idxT* __restrict Z_indices
) {
    idxT nnz = 0;
    Z_indptr[0] = 0;
    eT* Z_data_head = Z_data;
    idxT* Z_indices_head = Z_indices;
    const int n_mat = C_data.size();

    // threshold is already consistent between matrices, so accept every line.
    auto max_heap = MaxHeap<eT, idxT>(top_n, std::numeric_limits<eT>::min());

    // offset the index when concatenating the C sub-matrices (split by row)
    std::vector<idxT> offset(n_mat, idxT(0));
    for (int i = 0; i < n_mat - 1; ++i) {
        for (int j = i; j < n_mat - 1; ++j) {
            offset[j + 1] += B_ncols[i];
        }
    }

    // concatenate the results of each row, apply top_n and add those results to
    // the C matrix
    for (idxT i = 0; i < nrows; ++i) {
        eT min = max_heap.reset();

        // keep topn of stacked lines for each row insert in reverse order,
        // similar to the reverse linked list in sp_matmul_topn
        for (int j = n_mat - 1; j >= 0; --j) {
            const idxT* C_indptr_j = C_indptrs[j];
            const idxT* C_indices_j = C_indices[j];
            for (idxT k = C_indptr_j[i]; k < C_indptr_j[i + 1]; ++k) {
                eT val = (C_data[j])[k];
                if (val > min) {
                    min = max_heap.push_pop(offset[j] + C_indices_j[k], val);
                }
            }
        }

        // sort the heap s.t. the first value is the largest
        max_heap.value_sort();

        // fill the zipped sparse matrix Z
        int n_set = max_heap.get_n_set();
        for (int ii = 0; ii < n_set; ++ii) {
            *Z_indices_head = max_heap.heap[ii].idx;
            *Z_data_head = max_heap.heap[ii].val;
            Z_indices_head++;
            Z_data_head++;
        }
        nnz += n_set;
        Z_indptr[i + 1] = nnz;
    }
}

}  // namespace sdtn::core
