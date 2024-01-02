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
#include <sparse_dot_topn/sp_matmul_topn.hpp>
#include <sparse_dot_topn/sp_matmul_topn_bindings.hpp>

namespace sdtn {

namespace nb = nanobind;

namespace api {

template <typename eT>
using nb_vec
    = nb::ndarray<nb::numpy, eT, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename eT, typename idxT, core::iffInt<idxT> = true>
inline void sp_matmul_topn(
    const idxT top_n,
    const idxT nrows,
    const idxT ncols,
    const eT threshold,
    const nb_vec<eT>& A_data,
    const nb_vec<idxT>& A_indptr,
    const nb_vec<idxT>& A_indices,
    const nb_vec<eT>& B_data,
    const nb_vec<idxT>& B_indptr,
    const nb_vec<idxT>& B_indices,
    nb_vec<eT>& C_data,
    nb_vec<idxT>& C_indptr,
    nb_vec<idxT>& C_indices
) {
    core::sp_matmul_topn<eT, idxT>(
        top_n,
        nrows,
        ncols,
        threshold,
        A_data.data(),
        A_indptr.data(),
        A_indices.data(),
        B_data.data(),
        B_indptr.data(),
        B_indices.data(),
        C_data.data(),
        C_indptr.data(),
        C_indices.data()
    );
}

}  // namespace api

namespace bindings {

void bind_sp_matmul_topn(nb::module_& m);

}  // namespace bindings
}  // namespace sdtn
