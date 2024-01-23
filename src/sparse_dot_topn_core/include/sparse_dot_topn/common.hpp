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

#include <utility>
#include <vector>

namespace sdtn {
namespace core {

template <typename T>
using iffInt = std::enable_if_t<std::is_integral_v<T>, bool>;

}  // namespace core

namespace api {

namespace nb = nanobind;

template <typename eT>
using nb_vec
    = nb::ndarray<nb::numpy, eT, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename eT>
inline nb_vec<eT> to_nbvec(std::vector<eT>&& seq) {
    std::vector<eT>* seq_ptr = new std::vector<eT>(std::move(seq));
    eT* data = seq_ptr->data();
    auto capsule = nb::capsule(seq_ptr, [](void* p) noexcept {
        delete reinterpret_cast<std::vector<eT>*>(p);
    });
    return nb_vec<eT>(data, {seq_ptr->size()}, capsule);
}

template <typename eT>
inline nb_vec<eT> to_nbvec(eT* data, size_t size) {
    auto capsule = nb::capsule(data, [](void* p) noexcept {
        delete[] reinterpret_cast<eT*>(p);
    });
    return nb_vec<eT>(data, {size}, capsule);
}

}  // namespace api
}  // namespace sdtn
