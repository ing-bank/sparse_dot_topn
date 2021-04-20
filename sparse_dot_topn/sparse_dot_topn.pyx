#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at#
#    http://www.apache.org/licenses/LICENSE-2.0#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Zhe Sun, Ahmet Erdem
# April 20, 2017
# Modified by: Particular Miner
# April 14, 2021

# distutils: language = c++

from libcpp.vector cimport vector
from array_wrappers cimport ArrayWrapper_int, ArrayWrapper_double

cimport numpy as np
import numpy as np

np.import_array()


cdef extern from "sparse_dot_topn_source.h":

    cdef void sparse_dot_topn_source(
                                        int n_row,
                                        int n_col,
                                        int Ap[],
                                        int Aj[],
                                        double Ax[],
                                        int Bp[],
                                        int Bj[],
                                        double Bx[],
                                        int topn,
                                        double lower_bound,
                                        int Cp[],
                                        int Cj[],
                                        double Cx[]
                                    );

    cdef void sparse_dot_topn_extd_source(
                                        int n_row,
                                        int n_col,
                                        int Ap[],
                                        int Aj[],
                                        double Ax[],
                                        int Bp[],
                                        int Bj[],
                                        double Bx[],
                                        int topn,
                                        double lower_bound,
                                        int Cp[],
                                        int Cj[],
                                        double Cx[],
                                        int* nminmax
                                    );

    cdef void sparse_dot_free_source(
                                        int n_row,
                                        int n_col,
                                        int Ap[],
                                        int Aj[],
                                        double Ax[],
                                        int Bp[],
                                        int Bj[],
                                        double Bx[],
                                        double lower_bound,
                                        int Cp[],
                                        vector[int]* Cj,
                                        vector[double]* Cx,
                                        int* n_minmax
                                    );

    cdef void sparse_dot_only_max_nnz_col_source(
                                                    int n_row,
                                                    int n_col,
                                                    int Ap[],
                                                    int Aj[],
                                                    int Bp[],
                                                    int Bj[],
                                                    int* max_nnz_col
                                                );

cpdef sparse_dot_topn(
                        int n_row,
                        int n_col,
                        np.ndarray[int, ndim=1] a_indptr,
                        np.ndarray[int, ndim=1] a_indices,
                        np.ndarray[double, ndim=1] a_data,
                        np.ndarray[int, ndim=1] b_indptr,
                        np.ndarray[int, ndim=1] b_indices,
                        np.ndarray[double, ndim=1] b_data,
                        int ntop,
                        double lower_bound,
                        np.ndarray[int, ndim=1] c_indptr,
                        np.ndarray[int, ndim=1] c_indices,
                        np.ndarray[double, ndim=1] c_data
                    ):
    """
    Cython glue function to call sparse_dot_topn C++ implementation
    This function will return a matrix C in CSR format, where
    C = [sorted top n results and results > lower_bound for each row of A * B]

    Input:
        n_row: number of rows of A matrix
        n_col: number of columns of B matrix

        a_indptr, a_indices, a_data: CSR expression of A matrix
        b_indptr, b_indices, b_data: CSR expression of B matrix

        ntop: n top results
        lower_bound: a threshold that the element of A*B must greater than

    Output by reference:
        c_indptr, c_indices, c_data: CSR expression of C matrix

    N.B. A and B must be CSR format!!!
         The type of input numpy array must be aligned with types of C++ function arguments!
    """

    cdef int* Ap = &a_indptr[0]
    cdef int* Aj = &a_indices[0]
    cdef double* Ax = &a_data[0]
    cdef int* Bp = &b_indptr[0]
    cdef int* Bj = &b_indices[0]
    cdef double* Bx = &b_data[0]
    cdef int* Cp = &c_indptr[0]
    cdef int* Cj = &c_indices[0]
    cdef double* Cx = &c_data[0]

    sparse_dot_topn_source(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, ntop, lower_bound, Cp, Cj, Cx)
    return

cpdef sparse_dot_topn_extd(
                        int n_row,
                        int n_col,
                        np.ndarray[int, ndim=1] a_indptr,
                        np.ndarray[int, ndim=1] a_indices,
                        np.ndarray[double, ndim=1] a_data,
                        np.ndarray[int, ndim=1] b_indptr,
                        np.ndarray[int, ndim=1] b_indices,
                        np.ndarray[double, ndim=1] b_data,
                        int ntop,
                        double lower_bound,
                        np.ndarray[int, ndim=1] c_indptr,
                        np.ndarray[int, ndim=1] c_indices,
                        np.ndarray[double, ndim=1] c_data,
                        np.ndarray[int, ndim=1] nminmax,
                    ):
    """
    Cython glue function to call sparse_dot_topn C++ implementation
    This function will return a matrix C in CSR format, where
    C = [sorted top n results > lower_bound for each row of A * B]
    The maximum number of elements per row of C nminmax is also returned.

    Input:
        n_row: number of rows of A matrix
        n_col: number of columns of B matrix

        a_indptr, a_indices, a_data: CSR expression of A matrix
        b_indptr, b_indices, b_data: CSR expression of B matrix

        ntop: n top results
        lower_bound: a threshold that the element of A*B must greater than

    Output by reference:
        c_indptr, c_indices, c_data: CSR expression of C matrix
        nminmax: The maximum number of elements per row of C

    N.B. A and B must be CSR format!!!
         The type of input numpy array must be aligned with types of C++ function arguments!
    """

    cdef int* Ap = &a_indptr[0]
    cdef int* Aj = &a_indices[0]
    cdef double* Ax = &a_data[0]
    cdef int* Bp = &b_indptr[0]
    cdef int* Bj = &b_indices[0]
    cdef double* Bx = &b_data[0]
    cdef int* Cp = &c_indptr[0]
    cdef int* Cj = &c_indices[0]
    cdef double* Cx = &c_data[0]
    cdef int* n_minmax = &nminmax[0]

    sparse_dot_topn_extd_source(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, ntop, lower_bound, Cp, Cj, Cx, n_minmax)
    return

cpdef sparse_dot_free(
                        int n_row,
                        int n_col,
                        np.ndarray[int, ndim=1] a_indptr,
                        np.ndarray[int, ndim=1] a_indices,
                        np.ndarray[double, ndim=1] a_data,
                        np.ndarray[int, ndim=1] b_indptr,
                        np.ndarray[int, ndim=1] b_indices,
                        np.ndarray[double, ndim=1] b_data,
                        double lower_bound,
                        np.ndarray[int, ndim=1] c_indptr
                    ):
    """
    Cython glue function to call sparse_dot_free C++ implementation
    This function will return a matrix C in CSR format, where
    C = [all results > lower_bound for each row of A * B]
    This function lets C++ decide how to manage (grow/allocate/reallocate) memory for the 
    storage of these results as needed during the computation; then hands over to numpy
    a pointer to the memory location where the data resides  

    Input:
        n_row: number of rows of A matrix
        n_col: number of columns of B matrix

        a_indptr, a_indices, a_data: CSR expression of A matrix
        b_indptr, b_indices, b_data: CSR expression of B matrix

        lower_bound: a threshold that the element of A*B must greater than

    Output by reference:
        c_indptr, c_indices, c_data: CSR expression of C matrix

    N.B. A and B must be CSR format!!!
         The type of input numpy array must be aligned with types of C++ function arguments!
    """

    cdef int* Ap = &a_indptr[0]
    cdef int* Aj = &a_indices[0]
    cdef double* Ax = &a_data[0]
    cdef int* Bp = &b_indptr[0]
    cdef int* Bj = &b_indices[0]
    cdef double* Bx = &b_data[0]
    cdef int* Cp = &c_indptr[0]
    cdef np.ndarray[int, ndim=1] nminmax = np.array([0], dtype=np.int32)
    cdef int* n_minmax = &nminmax[0]
    
    cdef vector[int] vCj;
    cdef vector[double] vCx;

    sparse_dot_free_source(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, lower_bound, Cp, &vCj, &vCx, n_minmax)
    
    c_indices = np.asarray(ArrayWrapper_int(vCj)).squeeze(axis=0)
    c_data = np.asarray(ArrayWrapper_double(vCx)).squeeze(axis=0)
    
    return c_indices, c_data, nminmax[0]


cpdef sparse_dot_only_max_nnz_col(
                                    int n_row,
                                    int n_col,
                                    np.ndarray[int, ndim=1] a_indptr,
                                    np.ndarray[int, ndim=1] a_indices,
                                    np.ndarray[int, ndim=1] b_indptr,
                                    np.ndarray[int, ndim=1] b_indices,
                                    np.ndarray[int, ndim=1] minmax_topn
                                ):
    """
    Cython glue function to call sparse_dot_only_minmax_topn C++ implementation
    This function will return the maximum number of columns set
    per row over all rows of A * B

    Input:
        n_row: number of rows of A matrix
        n_col: number of columns of B matrix

        a_indptr, a_indices: CSR indices of A matrix
        b_indptr, b_indices: CSR indices of B matrix

    Output by reference:
        minmax_ntop: the maximum number of columns set per row over all rows of 
                     A * B

    N.B. A and B must be CSR format!!!
         The type of input numpy array must be aligned with types of C++ function arguments!
    """

    cdef int* Ap = &a_indptr[0]
    cdef int* Aj = &a_indices[0]
    cdef int* Bp = &b_indptr[0]
    cdef int* Bj = &b_indices[0]
    cdef int* o_minmax_topn = &minmax_topn[0]

    sparse_dot_only_max_nnz_col_source(n_row, n_col, Ap, Aj, Bp, Bj, o_minmax_topn)
    return
