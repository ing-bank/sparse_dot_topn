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
#include <nanobind/stl/optional.h>

#include <sparse_dot_topn/common.hpp>
#include <sparse_dot_topn/sp_matmul.hpp>

namespace sdtn {

namespace nb = nanobind;

namespace api {

template <typename eT, typename idxT, core::iffInt<idxT> = true>
inline nb::tuple sp_matmul(
    const idxT nrows,
    const idxT ncols,
    const nb_vec<eT>& A_data,
    const nb_vec<idxT>& A_indptr,
    const nb_vec<idxT>& A_indices,
    const nb_vec<eT>& B_data,
    const nb_vec<idxT>& B_indptr,
    const nb_vec<idxT>& B_indices
) {
    idxT* C_indptr = new idxT[nrows + 1];
    idxT result_size = core::sp_matmul_size(
        nrows,
        ncols,
        A_indptr.data(),
        A_indices.data(),
        B_indptr.data(),
        B_indices.data(),
        C_indptr
    );

    idxT* C_indices = new idxT[result_size];
    eT* C_data = new eT[result_size];

    core::sp_matmul<eT, idxT>(
        nrows,
        ncols,
        A_data.data(),
        A_indptr.data(),
        A_indices.data(),
        B_data.data(),
        B_indptr.data(),
        B_indices.data(),
        C_data,
        C_indices
    );
    return nb::make_tuple(
        to_nbvec<eT>(C_data, result_size),
        to_nbvec<idxT>(C_indices, result_size),
        to_nbvec<idxT>(C_indptr, nrows + 1)
    );
}

#if defined(SDTN_OMP_ENABLED)
template <typename eT, typename idxT, core::iffInt<idxT> = true>
inline nb::tuple sp_matmul_mt(
    const idxT nrows,
    const idxT ncols,
    const int n_threads,
    const nb_vec<eT>& A_data,
    const nb_vec<idxT>& A_indptr,
    const nb_vec<idxT>& A_indices,
    const nb_vec<eT>& B_data,
    const nb_vec<idxT>& B_indptr,
    const nb_vec<idxT>& B_indices
) {
    idxT* C_indptr = new idxT[nrows + 1];

    idxT result_size = core::sp_matmul_size_mt<idxT>(
        nrows,
        ncols,
        A_indptr.data(),
        A_indices.data(),
        B_indptr.data(),
        B_indices.data(),
        C_indptr
    );
    idxT* C_indices = new idxT[result_size];
    eT* C_data = new eT[result_size];

    core::sp_matmul_mt<eT, idxT>(
        nrows,
        ncols,
        n_threads,
        A_data.data(),
        A_indptr.data(),
        A_indices.data(),
        B_data.data(),
        B_indptr.data(),
        B_indices.data(),
        C_data,
        C_indptr,
        C_indices
    );
    return nb::make_tuple(
        to_nbvec<eT>(C_data, result_size),
        to_nbvec<idxT>(C_indices, result_size),
        to_nbvec<idxT>(C_indptr, nrows + 1)
    );
}
#endif  // SDTN_OMP_ENABLED

}  // namespace api

namespace bindings {

void bind_sp_matmul(nb::module_& m);
#if defined(SDTN_OMP_ENABLED)
void bind_sp_matmul_mt(nb::module_& m);
#endif  // SDTN_OMP_ENABLED

}  // namespace bindings
}  // namespace sdtn
