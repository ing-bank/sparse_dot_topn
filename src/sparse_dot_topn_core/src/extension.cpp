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
#include <sparse_dot_topn/sp_matmul_bindings.hpp>
#include <sparse_dot_topn/sp_matmul_topn_bindings.hpp>
#include <sparse_dot_topn/zip_sp_matmul_topn_bindings.hpp>

namespace sdtn::bindings {

NB_MODULE(_sparse_dot_topn_core, m) {
    bind_sp_matmul(m);
    bind_sp_matmul_topn(m);
    bind_sp_matmul_topn_sorted(m);
    bind_zip_sp_matmul_topn(m);
#ifdef SDTN_OMP_ENABLED
    bind_sp_matmul_mt(m);
    bind_sp_matmul_topn_mt(m);
    bind_sp_matmul_topn_sorted_mt(m);
    m.attr("_has_openmp_support") = true;
#else
    m.attr("_has_openmp_support") = false;
#endif  // MMU_HAS_OPENMP_SUPPORT
}

}  // namespace sdtn::bindings
