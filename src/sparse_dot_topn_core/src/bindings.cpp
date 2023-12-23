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
#include <sparse_dot_topn/sparse_dot_topn.hpp>
#include <sparse_dot_topn/sparse_dot_topn_bindings.hpp>

namespace sdtn::bindings {
namespace nb = nanobind;

using namespace nb::literals;

void bind_sparse_dot_topn(nb::module_& m) {
    m.def(
        "sparse_dot_topn",
        &api::sparse_dot_topn<float, int>,
        "top_n"_a,
        "nrows"_a,
        "ncols"_a,
        "threshold"_a,
        "A_data"_a,
        "A_indptr"_a,
        "A_indices"_a,
        "B_data"_a,
        "B_indptr"_a,
        "B_indices"_a,
        "C_data"_a,
        "C_indptr"_a,
        "C_indices"_a,
        nb::raw_doc("Compute sparse dot product and keep top n.\n"
                    "\n"
                    "Args:\n"
                    "    arg (float | int): Input value\n"
                    "\n"
                    "Returns:\n"
                    "    float | int: Result of the identity operation")
    );
}

}  // namespace sdtn::bindings
