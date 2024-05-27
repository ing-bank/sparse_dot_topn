/* sparse_dot_topn/maxheap.hpp -- Max heap implementation for Score structs.
 *
 * Copyright (c) 2023 ING Analytics Wholesale Banking
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
#include <algorithm>
#include <functional>
#include <limits>
#include <vector>

namespace sdtn::core {

template <typename eT, typename idxT>
struct Score {
    int order;
    idxT idx;
    eT val;
    bool operator>(const Score& other) const { return val > other.val; }
    bool operator<(const Score& other) const { return order < other.order; }
};

/**
 * \brief Container that retains top n values.
 *
 * \details MaxHeap implements MinMaxHeap that is sorted based on the values.
 *
 * \tparam eT   element type of the matrices
 * \tparam idxT integer type of the index arrays, must be at least 32 bit int
 */
template <typename eT, typename idxT>
class MaxHeap {
    using compare = std::greater<Score<eT, idxT>>;
    const int heap_size;
    int n_set = 0;
    static constexpr int max_order = std::numeric_limits<int>::max();
    eT init;

 public:
    std::vector<Score<eT, idxT>> heap;

    /**
     * \brief Instantiate the container.
     *
     * \param n       maximum number of values to store
     * \param initial initial `val` to set for all the entries
     */
    explicit MaxHeap(int n, eT initial) : heap_size{n}, init{initial} {
        heap.reserve(n + 1);
        for (int i = 0; i < heap_size; i++) {
            heap.push_back({max_order, -1, initial});
        }
        std::make_heap(heap.begin(), heap.end(), compare());
        std::pop_heap(heap.begin(), heap.end(), compare());
    }

    eT reset() {
        n_set = 0;
        for (int i = 0; i < heap_size; i++) {
            heap[i].order = max_order;
            heap[i].idx = -1;
            heap[i].val = init;
        }
        return init;
    }

    [[nodiscard]] int get_n_set() const { return std::min(heap_size, n_set); }

    /**
     * \brief Pop minimum value and store `val`.
     *
     * \param idx index of the value
     * \param val value to store
     */
    eT push_pop(const idxT idx, const eT val) {
        std::pop_heap(heap.begin(), heap.end(), compare());
        heap.back().order = n_set;
        heap.back().idx = idx;
        heap.back().val = val;
        n_set++;
        std::push_heap(heap.begin(), heap.end(), compare());
        return heap.front().val;
    }

    /**
     * \brief Sort the heap according to the insertion order.
     *
     * \details Note that calling `insertion_sort` invalidates the heap.
     * Calls should be followed by a call to `reset`.
     */
    void insertion_sort() {
        std::sort(heap.begin(), heap.end(), std::less<Score<eT, idxT>>());
    }

    /**
     * \brief Sort the heap according to values.
     *
     * \details Note that calling `value_sort` invalidates the heap.
     * Calls should be followed by a call to `reset`.
     */
    void value_sort() { std::sort(heap.begin(), heap.end(), compare()); }
};

}  // namespace sdtn::core
