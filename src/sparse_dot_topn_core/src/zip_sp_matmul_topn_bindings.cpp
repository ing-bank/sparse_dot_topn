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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <sparse_dot_topn/zip_sp_matmul_topn_bindings.hpp>

namespace sdtn::bindings {
namespace nb = nanobind;

using namespace nb::literals;

void bind_zip_sp_matmul_topn(nb::module_& m) {
    m.def(
        "zip_sp_matmul_topn",
        &api::zip_sp_matmul_topn<double, int>,
        "top_n"_a,
        "Z_max_nnz"_a,
        "nrows"_a,
        "B_ncols"_a,
        "data"_a.noconvert(),
        "indptr"_a.noconvert(),
        "indices"_a.noconvert(),
        ("Compute sparse dot product and keep top n.\n"
         "\n"
         "Args:\n"
         "    top_n (int): the number of results to retain\n"
         "    Z_max_nnz (int): the maximum number of non-zero values in Z\n"
         "    nrows (int): the number of rows in `A`\n"
         "    B_ncols (NDArray[int]): the number of columns in each block "
         "of `B`\n"
         "    data (list[NDArray[int | float]]): the non-zero elements of "
         "each C\n"
         "    indptr (list[NDArray[int]]): the row indices for each "
         "`C_data`\n"
         "    indices (list[NDArray[int]]): the column indices for each "
         "`C_data`\n"
         "\n"
         "Returns:\n"
         "    Z_data (NDArray[int | float]): the non-zero elements of Z\n"
         "    Z_indptr (NDArray[int]): the row indices for `Z_data`\n"
         "    Z_indices (NDArray[int]): the column indices for `Z_data`\n"
         "\n")
    );
    m.def(
        "zip_sp_matmul_topn",
        &api::zip_sp_matmul_topn<float, int>,
        "top_n"_a,
        "Z_max_nnz"_a,
        "nrows"_a,
        "B_ncols"_a,
        "data"_a.noconvert(),
        "indptr"_a.noconvert(),
        "indices"_a.noconvert()
    );
    m.def(
        "zip_sp_matmul_topn",
        &api::zip_sp_matmul_topn<double, int64_t>,
        "top_n"_a,
        "Z_max_nnz"_a,
        "nrows"_a,
        "B_ncols"_a,
        "data"_a.noconvert(),
        "indptr"_a.noconvert(),
        "indices"_a.noconvert()
    );
    m.def(
        "zip_sp_matmul_topn",
        &api::zip_sp_matmul_topn<float, int64_t>,
        "top_n"_a,
        "Z_max_nnz"_a,
        "nrows"_a,
        "B_ncols"_a,
        "data"_a.noconvert(),
        "indptr"_a.noconvert(),
        "indices"_a.noconvert()
    );
    m.def(
        "zip_sp_matmul_topn",
        &api::zip_sp_matmul_topn<int, int>,
        "top_n"_a,
        "Z_max_nnz"_a,
        "nrows"_a,
        "B_ncols"_a,
        "data"_a.noconvert(),
        "indptr"_a.noconvert(),
        "indices"_a.noconvert()
    );
    m.def(
        "zip_sp_matmul_topn",
        &api::zip_sp_matmul_topn<int64_t, int>,
        "top_n"_a,
        "Z_max_nnz"_a,
        "nrows"_a,
        "B_ncols"_a,
        "data"_a.noconvert(),
        "indptr"_a.noconvert(),
        "indices"_a.noconvert()
    );
    m.def(
        "zip_sp_matmul_topn",
        &api::zip_sp_matmul_topn<int, int64_t>,
        "top_n"_a,
        "Z_max_nnz"_a,
        "nrows"_a,
        "B_ncols"_a,
        "data"_a.noconvert(),
        "indptr"_a.noconvert(),
        "indices"_a.noconvert()
    );
    m.def(
        "zip_sp_matmul_topn",
        &api::zip_sp_matmul_topn<int64_t, int64_t>,
        "top_n"_a,
        "Z_max_nnz"_a,
        "nrows"_a,
        "B_ncols"_a,
        "data"_a.noconvert(),
        "indptr"_a.noconvert(),
        "indices"_a.noconvert()
    );
}

}  // namespace sdtn::bindings
