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
#include <sparse_dot_topn/sp_matmul.hpp>
#include <sparse_dot_topn/sp_matmul_bindings.hpp>

namespace sdtn::bindings {
namespace nb = nanobind;

using namespace nb::literals;

void bind_sp_matmul(nb::module_& m) {
    m.def(
        "sp_matmul",
        &api::sp_matmul<double, int>,
        "nrows"_a,
        "ncols"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert(),
        ("Compute sparse dot product and keep top n.\n"
         "\n"
         "Args:\n"
         "    nrows (int): the number of rows in `A`\n"
         "    ncols (int): the number of columns in `B`\n"
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
        "sp_matmul",
        &api::sp_matmul<float, int>,
        "nrows"_a,
        "ncols"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul",
        &api::sp_matmul<double, int64_t>,
        "nrows"_a,
        "ncols"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul",
        &api::sp_matmul<float, int64_t>,
        "nrows"_a,
        "ncols"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul",
        &api::sp_matmul<int, int>,
        "nrows"_a,
        "ncols"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul",
        &api::sp_matmul<int64_t, int>,
        "nrows"_a,
        "ncols"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul",
        &api::sp_matmul<int, int64_t>,
        "nrows"_a,
        "ncols"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul",
        &api::sp_matmul<int64_t, int64_t>,
        "nrows"_a,
        "ncols"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
}

#ifdef SDTN_OMP_ENABLED
void bind_sp_matmul_mt(nb::module_& m) {
    m.def(
        "sp_matmul_mt",
        &api::sp_matmul_mt<double, int>,
        "nrows"_a,
        "ncols"_a,
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
         "    nrows (int): the number of rows in `A`\n"
         "    ncols (int): the number of columns in `B`\n"
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
        "sp_matmul_mt",
        &api::sp_matmul_mt<float, int>,
        "nrows"_a,
        "ncols"_a,
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_mt",
        &api::sp_matmul_mt<double, int64_t>,
        "nrows"_a,
        "ncols"_a,
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_mt",
        &api::sp_matmul_mt<float, int64_t>,
        "nrows"_a,
        "ncols"_a,
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_mt",
        &api::sp_matmul_mt<int, int>,
        "nrows"_a,
        "ncols"_a,
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_mt",
        &api::sp_matmul_mt<int64_t, int>,
        "nrows"_a,
        "ncols"_a,
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_mt",
        &api::sp_matmul_mt<int, int64_t>,
        "nrows"_a,
        "ncols"_a,
        "n_threads"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert()
    );
    m.def(
        "sp_matmul_mt",
        &api::sp_matmul_mt<int64_t, int64_t>,
        "nrows"_a,
        "ncols"_a,
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
