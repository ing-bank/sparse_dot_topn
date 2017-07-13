# Created by Cool Kids in EM
# April 20, 2017

# distutils: language = c++

import numpy as np
cimport numpy as np

cdef extern from "cossim_topn_source.h":

    cdef void cossim_topn_source(
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
                        double Cx[]);

cpdef cossim_topn(
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
    Cython glue function to call cossim_topn C++ implementation
    This function will return a matrxi C in CSR format, where
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
         The type of input numpy array must be aligned with types of C++ function aguments!
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

    cossim_topn_source(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, ntop, lower_bound, Cp, Cj, Cx)
    return
