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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <numeric>
#include <vector>

#include <sparse_dot_topn/common.hpp>
#include <sparse_dot_topn/maxheap.hpp>
#include <sparse_dot_topn/zip_sp_matmul_topn.hpp>

namespace sdtn {
namespace nb = nanobind;

namespace api {

template <typename eT, typename idxT, core::iffInt<idxT> = true>
inline nb::tuple zip_sp_matmul_topn(
    const int top_n,
    const idxT Z_max_nnz,
    const idxT nrows,
    const nb_vec<idxT>& B_ncols,
    const std::vector<nb_vec<eT>>& data,
    const std::vector<nb_vec<idxT>>& indptr,
    const std::vector<nb_vec<idxT>>& indices
) {
    const int n_mats = B_ncols.size();
    std::vector<const eT*> data_ptrs;
    data_ptrs.reserve(n_mats);
    std::vector<const idxT*> indptr_ptrs;
    indptr_ptrs.reserve(n_mats);
    std::vector<const idxT*> indices_ptrs;
    indices_ptrs.reserve(n_mats);

    for (int i = 0; i < n_mats; ++i) {
        data_ptrs.push_back(data[i].data());
        indptr_ptrs.push_back(indptr[i].data());
        indices_ptrs.push_back(indices[i].data());
    }

    auto Z_indptr = std::unique_ptr<idxT[]>(new idxT[nrows + 1]);
    auto Z_indices = std::unique_ptr<idxT>(new idxT[Z_max_nnz]);
    auto Z_data = std::unique_ptr<eT>(new eT[Z_max_nnz]);

    core::zip_sp_matmul_topn<eT, idxT>(
        top_n,
        nrows,
        B_ncols.data(),
        data_ptrs,
        indptr_ptrs,
        indices_ptrs,
        Z_data.get(),
        Z_indptr.get(),
        Z_indices.get()
    );

    return nb::make_tuple(
        to_nbvec<eT>(Z_data.release(), Z_max_nnz),
        to_nbvec<idxT>(Z_indices.release(), Z_max_nnz),
        to_nbvec<idxT>(Z_indptr.release(), nrows + 1)
    );
}
}  //  namespace api

namespace bindings {
void bind_zip_sp_matmul_topn(nb::module_& m);
}

}  // namespace sdtn
