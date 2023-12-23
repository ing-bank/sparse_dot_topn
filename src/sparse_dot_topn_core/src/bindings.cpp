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
#include <sparse_dot_topn/sparse_dot_topn_bindings.hpp>
namespace sdtn::bindings {

namespace nb = nanobind;
using namespace nb::literals;

void bind_add(nb::module_& m) {
    m.def(
        "add", [](int a, int b) { return a + b; }, "a"_a, "b"_a
    );
}

}  // namespace sdtn::bindings
