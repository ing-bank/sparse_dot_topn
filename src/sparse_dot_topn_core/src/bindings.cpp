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
#include <nanobind/ndarray.h>
#include <sparse_dot_topn/topn_sp_matmul.hpp>
#include <sparse_dot_topn/topn_sp_matmul_bindings.hpp>

namespace sdtn::bindings {
namespace nb = nanobind;

using namespace nb::literals;

void bind_topn_sp_matmul(nb::module_& m) {
    m.def(
        "topn_sp_matmul",
        &api::topn_sp_matmul<double, int>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert(),
        "C_data"_a.noconvert(),
        "C_indptr"_a.noconvert(),
        "C_indices"_a.noconvert(),
        nb::raw_doc(
            "Compute sparse dot product and keep top n.\n"
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
            "    C_data (NDArray[int | float]): the non-zero elements of C\n"
            "    C_indptr (NDArray[int]): the row indices for `C_data`\n"
            "    C_indices (NDArray[int]): the column indices for `C_data`\n"
            "\n"
        )
    );
    m.def(
        "topn_sp_matmul",
        &api::topn_sp_matmul<float, int>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert(),
        "C_data"_a.noconvert(),
        "C_indptr"_a.noconvert(),
        "C_indices"_a.noconvert()
    );
    m.def(
        "topn_sp_matmul",
        &api::topn_sp_matmul<double, int64_t>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert(),
        "C_data"_a.noconvert(),
        "C_indptr"_a.noconvert(),
        "C_indices"_a.noconvert()
    );
    m.def(
        "topn_sp_matmul",
        &api::topn_sp_matmul<float, int64_t>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a,
        "A_data"_a.noconvert(),
        "A_indptr"_a.noconvert(),
        "A_indices"_a.noconvert(),
        "B_data"_a.noconvert(),
        "B_indptr"_a.noconvert(),
        "B_indices"_a.noconvert(),
        "C_data"_a.noconvert(),
        "C_indptr"_a.noconvert(),
        "C_indices"_a.noconvert()
    );
}

}  // namespace sdtn::bindings
