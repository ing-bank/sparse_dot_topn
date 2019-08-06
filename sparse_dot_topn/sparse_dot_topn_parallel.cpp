/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Author: Zhe Sun, Ahmet Erdem
// April 20, 2017

#include <vector>
#include <limits>
#include <algorithm>
#include <thread>

#include "./sparse_dot_topn_source.h"
#include "./sparse_dot_topn_parallel.h"

void inner_sparse_function(int start_row, int end_row, int n_col_inner,
                            int ntop_inner, double lower_bound_inner, int Ap_copy[],
                            int Aj_copy[], double Ax_copy[], int Bp_copy[], int Bj_copy[],
                            double Bx_copy[], std::vector<candidate> real_candidates[])
{

std::vector<int> next(n_col_inner,-1);
std::vector<double> sums(n_col_inner, 0);

std::vector<candidate> temp_candidates;

int iterations_count = 0;

for(int i = start_row; i < end_row; i++){

    iterations_count += 1;

    int head   = -2;
    int length =  0;

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
            temp_candidates.push_back(c);
        }

        int temp = head;
        head = next[head]; //iterate over columns

        next[temp] = -1; //clear arrays
        sums[temp] =  0; //clear arrays
    }

    int len = (int)temp_candidates.size();
    if (len > ntop_inner){
        std::partial_sort(temp_candidates.begin(),
                            temp_candidates.begin()+ntop_inner,
                            temp_candidates.end(),
                            candidate_cmp);
        len = ntop_inner;
    } else {
        std::sort(temp_candidates.begin(),
                    temp_candidates.end(), candidate_cmp);
    }


    temp_candidates.resize(len);
    real_candidates[i] = temp_candidates;

    temp_candidates.clear();

}

}

void sparse_dot_topn_parallel(int n_row,
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
                        int n_jobs)
{

    Cp[0] = 0;

	int split_amount = n_row / n_jobs;

	std::vector<std::vector<int>> split_row_vector(n_jobs);

	std::vector<std::vector<candidate>> real_candidates(n_row);

	std::vector<candidate> *real_cand_pointer;
	real_cand_pointer = &real_candidates[0];

	std::vector<std::thread> thread_list(n_jobs);


	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {
	    std::vector<int> temp_vector(2, 0);

	    int start_split = job_nr * split_amount;
	    int end_split = start_split + split_amount;

	    if (job_nr == n_jobs -1) {
	        end_split = n_row;
	    }

	    temp_vector[0] = start_split;
	    temp_vector[1] = end_split;

	    split_row_vector[job_nr] = temp_vector;

	}


	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {


	    int start_row = split_row_vector[job_nr][0];
	    int end_row = split_row_vector[job_nr][1];


	    thread_list[job_nr] = std::thread (inner_sparse_function, start_row,
	                                        end_row, n_col, ntop, lower_bound,
	                                        Ap, Aj, Ax, Bp, Bj, Bx,
	                                        real_cand_pointer);

    }

    for (int job_nr = 0; job_nr < n_jobs; job_nr++) {
        thread_list[job_nr].join();
    }

    int nnz = 0;

    for (int m = 0; m < n_row; m++) {

        std::vector<candidate> cand = real_cand_pointer[m];

        int can_len = (int)cand.size();

        for(int can_nr=0; can_nr < can_len; can_nr++){
            Cj[nnz] = cand[can_nr].index;
            Cx[nnz] = cand[can_nr].value;
            nnz++;
        }

        Cp[m+1] = nnz;

    }

}
