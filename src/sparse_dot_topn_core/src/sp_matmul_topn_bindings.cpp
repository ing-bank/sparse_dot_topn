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
#include <sparse_dot_topn/sp_matmul_topn.hpp>
#include <sparse_dot_topn/sp_matmul_topn_bindings.hpp>

namespace sdtn::bindings {
namespace nb = nanobind;

using namespace nb::literals;

void bind_sp_matmul_topn(nb::module_& m) {
    m.def(
        "sp_matmul_topn",
        &api::sp_matmul_topn<double, int, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert(),
        ("Compute sparse dot product and keep top n.\n"
         "\n"
         "Args:\n"
         "    top_n (int): the number of results to retain\n"
         "    nrows (int): the number of rows in `A`\n"
         "    ncols (int): the number of columns in `B`\n"
         "    density (float): the expected density of the result"
         " considering `top_n`\n"
         "    threshold (float): only store values greater than\n"
         "    A_data (NDArray[int | float]): the non-zero elements of A\n"
         "    A_indptr (NDArray[int]): the row indices for `A_data`\n"
         "    A_indices (NDArray[int]): the column indices for `A_data`\n"
         "    B_data (NDArray[int | float]): the non-zero elements of B\n"
         "    B_indptr (NDArray[int]): the row indices for `B_data`\n"
         "    B_indices (NDArray[int]): the column indices for `B_data`\n"
         "\n"
         "Returns:\n"
         "    C_data (NDArray[int | float]): the non-zero elements of C\n"
         "    C_indptr (NDArray[int]): the row indices for `C_data`\n"
         "    C_indices (NDArray[int]): the column indices for `C_data`\n"
         "\n")
    );
    m.def(
        "sp_matmul_topn",
        &api::sp_matmul_topn<float, int, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn",
        &api::sp_matmul_topn<double, int64_t, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn",
        &api::sp_matmul_topn<float, int64_t, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn",
        &api::sp_matmul_topn<int, int, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn",
        &api::sp_matmul_topn<int64_t, int, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn",
        &api::sp_matmul_topn<int, int64_t, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn",
        &api::sp_matmul_topn<int64_t, int64_t, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
}

void bind_sp_matmul_topn_sorted(nb::module_& m) {
    m.def(
        "sp_matmul_topn_sorted",
        &api::sp_matmul_topn<double, int, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert(),
        ("Compute sparse dot product and keep top n.\n"
         "\n"
         "Args:\n"
         "    top_n (int): the number of results to retain\n"
         "    nrows (int): the number of rows in `A`\n"
         "    ncols (int): the number of columns in `B`\n"
         "    threshold (float): only store values greater than\n"
         "    density (float): the expected density of the result"
         " considering `top_n`\n"
         "    A_data (NDArray[int | float]): the non-zero elements of A\n"
         "    A_indptr (NDArray[int]): the row indices for `A_data`\n"
         "    A_indices (NDArray[int]): the column indices for `A_data`\n"
         "    B_data (NDArray[int | float]): the non-zero elements of B\n"
         "    B_indptr (NDArray[int]): the row indices for `B_data`\n"
         "    B_indices (NDArray[int]): the column indices for `B_data`\n"
         "\n"
         "Returns:\n"
         "    C_data (NDArray[int | float]): the non-zero elements of C\n"
         "    C_indptr (NDArray[int]): the row indices for `C_data`\n"
         "    C_indices (NDArray[int]): the column indices for `C_data`\n"
         "\n")
    );
    m.def(
        "sp_matmul_topn_sorted",
        &api::sp_matmul_topn<float, int, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted",
        &api::sp_matmul_topn<double, int64_t, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted",
        &api::sp_matmul_topn<float, int64_t, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted",
        &api::sp_matmul_topn<int, int, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted",
        &api::sp_matmul_topn<int64_t, int, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted",
        &api::sp_matmul_topn<int, int64_t, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted",
        &api::sp_matmul_topn<int64_t, int64_t, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "density"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
}

#ifdef SDTN_OMP_ENABLED
void bind_sp_matmul_topn_mt(nb::module_& m) {
    m.def(
        "sp_matmul_topn_mt",
        &api::sp_matmul_topn_mt<double, int, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert(),
        ("Compute sparse dot product and keep top n.\n"
         "\n"
         "Args:\n"
         "    top_n (int): the number of results to retain\n"
         "    nrows (int): the number of rows in `A`\n"
         "    ncols (int): the number of columns in `B`\n"
         "    threshold (float): only store values greater than\n"
         "    A_data (NDArray[int | float]): the non-zero elements of A\n"
         "    A_indptr (NDArray[int]): the row indices for `A_data`\n"
         "    A_indices (NDArray[int]): the column indices for `A_data`\n"
         "    B_data (NDArray[int | float]): the non-zero elements of B\n"
         "    B_indptr (NDArray[int]): the row indices for `B_data`\n"
         "    B_indices (NDArray[int]): the column indices for `B_data`\n"
         "\n"
         "Returns:\n"
         "    C_data (NDArray[int | float]): the non-zero elements of C\n"
         "    C_indptr (NDArray[int]): the row indices for `C_data`\n"
         "    C_indices (NDArray[int]): the column indices for `C_data`\n"
         "\n")
    );
    m.def(
        "sp_matmul_topn_mt",
        &api::sp_matmul_topn_mt<float, int, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_mt",
        &api::sp_matmul_topn_mt<double, int64_t, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_mt",
        &api::sp_matmul_topn_mt<float, int64_t, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_mt",
        &api::sp_matmul_topn_mt<int, int, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_mt",
        &api::sp_matmul_topn_mt<int64_t, int, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_mt",
        &api::sp_matmul_topn_mt<int, int64_t, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_mt",
        &api::sp_matmul_topn_mt<int64_t, int64_t, true>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
}

void bind_sp_matmul_topn_sorted_mt(nb::module_& m) {
    m.def(
        "sp_matmul_topn_sorted_mt",
        &api::sp_matmul_topn_mt<double, int, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert(),
        ("Compute sparse dot product and keep top n.\n"
         "\n"
         "Args:\n"
         "    top_n (int): the number of results to retain\n"
         "    nrows (int): the number of rows in `A`\n"
         "    ncols (int): the number of columns in `B`\n"
         "    threshold (float): only store values greater than\n"
         "    A_data (NDArray[int | float]): the non-zero elements of A\n"
         "    A_indptr (NDArray[int]): the row indices for `A_data`\n"
         "    A_indices (NDArray[int]): the column indices for `A_data`\n"
         "    B_data (NDArray[int | float]): the non-zero elements of B\n"
         "    B_indptr (NDArray[int]): the row indices for `B_data`\n"
         "    B_indices (NDArray[int]): the column indices for `B_data`\n"
         "\n"
         "Returns:\n"
         "    C_data (NDArray[int | float]): the non-zero elements of C\n"
         "    C_indptr (NDArray[int]): the row indices for `C_data`\n"
         "    C_indices (NDArray[int]): the column indices for `C_data`\n"
         "\n")
    );
    m.def(
        "sp_matmul_topn_sorted_mt",
        &api::sp_matmul_topn_mt<float, int, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted_mt",
        &api::sp_matmul_topn_mt<double, int64_t, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted_mt",
        &api::sp_matmul_topn_mt<float, int64_t, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted_mt",
        &api::sp_matmul_topn_mt<int, int, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted_mt",
        &api::sp_matmul_topn_mt<int64_t, int, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted_mt",
        &api::sp_matmul_topn_mt<int, int64_t, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_topn_sorted_mt",
        &api::sp_matmul_topn_mt<int64_t, int64_t, false>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a.none(),
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
}
#endif  // SDTN_OMP_ENABLED

}  // namespace sdtn::bindings
