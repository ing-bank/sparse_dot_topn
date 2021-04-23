/*
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

// Author: Zhe Sun, Ahmet Erdem
// April 20, 2017
// Modified by: Particular Miner
// April 14, 2021

#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <thread>
#include <iostream>

#include "./sparse_dot_topn_source.h"
#include "./sparse_dot_topn_parallel.h"

struct job_range_type {int begin; int end;};

void distribute_load(
		int load_sz,
		int n_jobs,
		std::vector<job_range_type> &ranges
)
{
	// share the load among jobs:
	int equal_job_load_sz = load_sz/n_jobs;
	int rem = load_sz % n_jobs;
	ranges.resize(n_jobs);

	int start = 0;
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		ranges[job_nr].begin = start;
		ranges[job_nr].end = start + equal_job_load_sz + ((job_nr < rem)? 1 : 0);
		start = ranges[job_nr].end;
	}
}

void inner_gather_function(
		job_range_type job_range,
		int Cp[],
		int Cp_start,
		int vCj_start[],
		double vCx_start[],
		std::vector<candidate>* real_candidates,
		std::vector<int>* row_sizes
)
{
	candidate* c = real_candidates->data();
	int* vCj_cursor = &vCj_start[Cp_start];
	double* vCx_cursor = &vCx_start[Cp_start];

	int Cp_i = Cp_start;
	int* row_sizes_ptr = row_sizes->data();

	for (int i = job_range.begin; i < job_range.end; i++){
		for (int j = 0; j < (*row_sizes_ptr); j++){
			*(vCj_cursor++) = c->index;
			*(vCx_cursor++) = (c++)->value;
		}
		Cp_i += *(row_sizes_ptr++);
		Cp[i + 1] = Cp_i;
	}
	real_candidates->clear();
}

void inner_sparse_dot_topn(
		job_range_type job_range,
		int n_col_inner,
		int ntop_inner,
		double lower_bound_inner,
		int Ap_copy[],
		int Aj_copy[],
		double Ax_copy[],
		int Bp_copy[],
		int Bj_copy[],
		double Bx_copy[],
		std::vector<candidate>* real_candidates,
		std::vector<int>* row_sizes,
		int* total
)
{
	std::vector<int> next(n_col_inner,-1);
	std::vector<double> sums(n_col_inner, 0);

	real_candidates->reserve(job_range.end - job_range.begin);

	row_sizes->resize(job_range.end - job_range.begin);
	int* row_sizes_ptr = row_sizes->data();

	for (int i = job_range.begin; i < job_range.end; i++){

		int head   = -2;
		int length =  0;
		size_t sz = real_candidates->size();

		int jj_start = Ap_copy[i];
		int jj_end   = Ap_copy[i+1];

		for(int jj = jj_start; jj < jj_end; jj++){
			int j = Aj_copy[jj];
			double v = Ax_copy[jj]; //value of A in (i,j)

			int kk_start = Bp_copy[j];
			int kk_end   = Bp_copy[j+1];
			for(int kk = kk_start; kk < kk_end; kk++){
				int k = Bj_copy[kk]; //kth column of B in row j

				sums[k] += v*Bx_copy[kk]; //multiply with value of B in (j,k) and accumulate to the result for kth column of row i

				if(next[k] == -1){
					next[k] = head; //keep a linked list, every element points to the next column index
					head  = k;
					length++;
				}
			}
		}

		for(int jj = 0; jj < length; jj++){ //length = number of columns set (may include 0s)

			if(sums[head] > lower_bound_inner){ //append the nonzero elements
				candidate c;
				c.index = head;
				c.value = sums[head];
				real_candidates->push_back(c);
			}

			int temp = head;
			head = next[head]; //iterate over columns

			next[temp] = -1; //clear arrays
			sums[temp] =  0; //clear arrays
		}

		int len = (int) (real_candidates->size() - sz);

		candidate* candidate_arr_begin = real_candidates->data() + sz;
		if (len > ntop_inner){
			std::partial_sort(
					candidate_arr_begin,
					candidate_arr_begin + ntop_inner,
					candidate_arr_begin + len,
					candidate_cmp
			);
			len = ntop_inner;
		}
		else {
			std::sort(
					candidate_arr_begin,
					candidate_arr_begin + len,
					candidate_cmp
			);
		}

		real_candidates->resize(sz + (size_t) len);
		*(row_sizes_ptr++) = len;
		(*total) += len;
	}
	real_candidates->shrink_to_fit();
}

void sparse_dot_topn_parallel(
		int n_row,
		int n_col,
		int Ap[],
		int Aj[],
		double Ax[], //data of A
		int Bp[],
		int Bj[],
		double Bx[], //data of B
		int ntop,
		double lower_bound,
		int Cp[],
		int Cj[],
		double Cx[],
		int n_jobs
)
{
	std::vector<job_range_type> job_ranges(n_jobs);
	distribute_load(n_row, n_jobs, job_ranges);

	std::vector<std::vector<candidate> > real_candidates(n_jobs);
	std::vector<std::vector<int>> row_sizes(n_jobs);

	// initialize aggregate:
	std::vector<int> sub_total(n_jobs, 0);

	std::vector<std::thread> thread_list(n_jobs);
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = std::thread(
				inner_sparse_dot_topn,
				job_ranges[job_nr],
				n_col, ntop,
				lower_bound,
				Ap, Aj, Ax, Bp, Bj, Bx,
				&real_candidates[job_nr],
				&row_sizes[job_nr],
				&sub_total[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

	// gather the results:
	std::vector<int> start_points(n_jobs + 1);
	start_points[0] = 0;
	partial_sum(sub_total.begin(), sub_total.end(), start_points.begin() + 1);

	Cp[0] = 0;
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = std::thread(
				inner_gather_function,
				job_ranges[job_nr],
				Cp,
				start_points[job_nr],
				Cj,
				Cx,
				&real_candidates[job_nr],
				&row_sizes[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

}

void inner_sparse_dot_topn_extd(
		job_range_type job_range,
		int n_col_inner,
		int ntop_inner,
		double lower_bound_inner,
		int Ap_copy[],
		int Aj_copy[],
		double Ax_copy[],
		int Bp_copy[],
		int Bj_copy[],
		double Bx_copy[],
		std::vector<candidate>* real_candidates,
		std::vector<int>* row_sizes,
		int* total,
		int* n_minmax
)
{
	std::vector<int> next(n_col_inner,-1);
	std::vector<double> sums(n_col_inner, 0);

	real_candidates->reserve(job_range.end - job_range.begin);

	row_sizes->resize(job_range.end - job_range.begin);
	int* row_sizes_ptr = row_sizes->data();

	for(int i = job_range.begin; i < job_range.end; i++){

		int head   = -2;
		int length =  0;
		size_t sz = real_candidates->size();

		int jj_start = Ap_copy[i];
		int jj_end   = Ap_copy[i+1];

		for(int jj = jj_start; jj < jj_end; jj++){
			int j = Aj_copy[jj];
			double v = Ax_copy[jj]; //value of A in (i,j)

			int kk_start = Bp_copy[j];
			int kk_end   = Bp_copy[j+1];
			for(int kk = kk_start; kk < kk_end; kk++){
				int k = Bj_copy[kk]; //kth column of B in row j

				sums[k] += v*Bx_copy[kk]; //multiply with value of B in (j,k) and accumulate to the result for kth column of row i

				if(next[k] == -1){
					next[k] = head; //keep a linked list, every element points to the next column index
					head  = k;
					length++;
				}
			}
		}

		for(int jj = 0; jj < length; jj++){ //length = number of columns set (may include 0s)

			if(sums[head] > lower_bound_inner){ //append the nonzero elements
				candidate c;
				c.index = head;
				c.value = sums[head];
				real_candidates->push_back(c);
			}

			int temp = head;
			head = next[head]; //iterate over columns

			next[temp] = -1; //clear arrays
			sums[temp] =  0; //clear arrays
		}

		int len = (int) (real_candidates->size() - sz);
		*n_minmax = (len > *n_minmax)? len : *n_minmax;

		candidate* candidate_arr_begin = real_candidates->data() + sz;
		if (len > ntop_inner){
			std::partial_sort(
					candidate_arr_begin,
					candidate_arr_begin + ntop_inner,
					candidate_arr_begin + len,
					candidate_cmp
			);
			len = ntop_inner;
		}
		else {
			std::sort(
					candidate_arr_begin,
					candidate_arr_begin + len,
					candidate_cmp
			);
		}

		real_candidates->resize(sz + (size_t) len);
		*(row_sizes_ptr++) = len;
		(*total) += len;
	}
	real_candidates->shrink_to_fit();
}

void sparse_dot_topn_extd_parallel(
		int n_row,
		int n_col,
		int Ap[],
		int Aj[],
		double Ax[], //data of A
		int Bp[],
		int Bj[],
		double Bx[], //data of B
		int ntop,
		double lower_bound,
		int Cp[],
		int Cj[],
		double Cx[],
		int *n_minmax,
		int n_jobs
)
{
	std::vector<job_range_type> job_ranges(n_jobs);
	distribute_load(n_row, n_jobs, job_ranges);

	std::vector<std::vector<candidate> > real_candidates(n_jobs);
	std::vector<std::vector<int>> row_sizes(n_jobs);

	// initialize aggregates:
	std::vector<int> sub_total(n_jobs, 0);
	std::vector<int> split_n_minmax(n_jobs, 0);

	std::vector<std::thread> thread_list(n_jobs);

	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = std::thread(
				inner_sparse_dot_topn_extd,
				job_ranges[job_nr],
				n_col, ntop,
				lower_bound,
				Ap, Aj, Ax, Bp, Bj, Bx,
				&real_candidates[job_nr],
				&row_sizes[job_nr],
				&sub_total[job_nr],
				&split_n_minmax[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

	// gather the results:
	*n_minmax = *max_element(split_n_minmax.begin(), split_n_minmax.end());

	std::vector<int> start_points(n_jobs + 1);
	start_points[0] = 0;
	partial_sum(sub_total.begin(), sub_total.end(), start_points.begin() + 1);

	Cp[0] = 0;
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = std::thread(
				inner_gather_function,
				job_ranges[job_nr],
				Cp,
				start_points[job_nr],
				Cj,
				Cx,
				&real_candidates[job_nr],
				&row_sizes[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

}

void inner_sparse_dot_free(
		job_range_type job_range,
		int n_col_inner,
		double lower_bound_inner,
		int Ap_copy[],
		int Aj_copy[],
		double Ax_copy[],
		int Bp_copy[],
		int Bj_copy[],
		double Bx_copy[],
		std::vector<candidate>* real_candidates,
		std::vector<int>* row_sizes,
		int* total,
		int* n_minmax
)
{
	std::vector<int> next(n_col_inner,-1);
	std::vector<double> sums(n_col_inner, 0);

	real_candidates->reserve(job_range.end - job_range.begin);

	row_sizes->resize(job_range.end - job_range.begin);
	int* row_sizes_ptr = row_sizes->data();

	for(int i = job_range.begin; i < job_range.end; i++){

		int head   = -2;
		int length =  0;
		size_t sz = real_candidates->size();

		int jj_start = Ap_copy[i];
		int jj_end   = Ap_copy[i+1];

		for(int jj = jj_start; jj < jj_end; jj++){
			int j = Aj_copy[jj];
			double v = Ax_copy[jj]; //value of A in (i,j)

			int kk_start = Bp_copy[j];
			int kk_end   = Bp_copy[j+1];
			for(int kk = kk_start; kk < kk_end; kk++){
				int k = Bj_copy[kk]; //kth column of B in row j

				sums[k] += v*Bx_copy[kk]; //multiply with value of B in (j,k) and accumulate to the result for kth column of row i

				if(next[k] == -1){
					next[k] = head; //keep a linked list, every element points to the next column index
					head  = k;
					length++;
				}
			}
		}

		for(int jj = 0; jj < length; jj++){ //length = number of columns set (may include 0s)

			if(sums[head] > lower_bound_inner){ //append the nonzero elements
				candidate c;
				c.index = head;
				c.value = sums[head];
				real_candidates->push_back(c);
			}

			int temp = head;
			head = next[head]; //iterate over columns

			next[temp] = -1; //clear arrays
			sums[temp] =  0; //clear arrays
		}

		int len = (int) (real_candidates->size() - sz);

		candidate* candidate_arr_begin = real_candidates->data() + sz;
		std::sort(
				candidate_arr_begin,
				candidate_arr_begin + len,
				candidate_cmp
		);

		real_candidates->resize(sz + (size_t) len);
		*(row_sizes_ptr++) = len;
		(*total) += len;
		*n_minmax = (len > *n_minmax)? len : *n_minmax;
	}
	real_candidates->shrink_to_fit();
}

void sparse_dot_free_parallel(
		int n_row,
		int n_col,
		int Ap[],
		int Aj[],
		double Ax[], //data of A
		int Bp[],
		int Bj[],
		double Bx[], //data of B
		double lower_bound,
		int Cp[],
		std::vector<int>* vCj,
		std::vector<double>* vCx,
		int* n_minmax,
		int n_jobs
)
{
	std::vector<job_range_type> job_ranges(n_jobs);
	distribute_load(n_row, n_jobs, job_ranges);

	std::vector<std::vector<candidate> > real_candidates(n_jobs);
	std::vector<std::vector<int>> row_sizes(n_jobs);

	// initialize aggregates:
	std::vector<int> sub_total(n_jobs, 0);
	std::vector<int> split_n_minmax(n_jobs, 0);

	// execute the jobs:
	std::vector<std::thread> thread_list(n_jobs);
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = std::thread (
				inner_sparse_dot_free,
				job_ranges[job_nr],
				n_col,
				lower_bound,
				Ap, Aj, Ax, Bp, Bj, Bx,
				&real_candidates[job_nr],
				&row_sizes[job_nr],
				&sub_total[job_nr],
				&split_n_minmax[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

	// gather the results (in parallel):
	*n_minmax = *std::max_element(split_n_minmax.begin(), split_n_minmax.end());

	std::vector<int> start_points(n_jobs + 1);
	start_points[0] = 0;
	std::partial_sum(sub_total.begin(), sub_total.end(), start_points.begin() + 1);

	int total = start_points.back();
	vCj->resize(total);
	vCj->shrink_to_fit();
	vCx->resize(total);
	vCx->shrink_to_fit();

	Cp[0] = 0;
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = std::thread(
				inner_gather_function,
				job_ranges[job_nr],
				Cp,
				start_points[job_nr],
				&((*vCj)[0]),
				&((*vCx)[0]),
				&real_candidates[job_nr],
				&row_sizes[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

}

void inner_sparse_only_max_nnz_col(
		job_range_type job_range,
		int n_col_inner,
		int Ap_copy[],
		int Aj_copy[],
		int Bp_copy[],
		int Bj_copy[],
		int *max_nnz_col	// already initialized to 0
)
{
	std::vector<bool> unmarked(n_col_inner, true);

	for(int i = job_range.begin; i < job_range.end; i++){

		int length =  0;

		int jj_start = Ap_copy[i];
		int jj_end   = Ap_copy[i+1];

		for(int jj = jj_start; jj < jj_end; jj++){
			int j = Aj_copy[jj];

			int kk_start = Bp_copy[j];
			int kk_end   = Bp_copy[j+1];
			for(int kk = kk_start; kk < kk_end; kk++){
				int k = Bj_copy[kk]; //kth column of B in row j

				if(unmarked[k]){	// if this k is not already marked then ...
					unmarked[k] = false;	// keep a record of column k
					length++;
				}
			}
		}
		*max_nnz_col = (length > *max_nnz_col)? length : *max_nnz_col;
	}
}

void sparse_dot_only_max_nnz_col_parallel(
		int n_row,
		int n_col,
		int Ap[],
		int Aj[],
		int Bp[],
		int Bj[],
		int *max_nnz_col,
		int n_jobs
)
{
	std::vector<job_range_type> job_ranges(n_jobs);
	distribute_load(n_row, n_jobs, job_ranges);

	std::vector<int> split_max_nnz_col(n_jobs, 0);
	std::vector<std::thread> thread_list(n_jobs);
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = std::thread (
				inner_sparse_only_max_nnz_col,
				job_ranges[job_nr],
				n_col,
				Ap, Aj, Bp, Bj,
				&split_max_nnz_col[job_nr]
		);

	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

	*max_nnz_col = *max_element(split_max_nnz_col.begin(), split_max_nnz_col.end());
}
