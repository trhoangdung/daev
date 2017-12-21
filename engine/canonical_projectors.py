'''
This module implements canonical projectors and its related functions used to decompose the DAE equation
Dung Tran: Dec/2017
'''

from scipy.sparse import issparse, random
from scipy.sparse.linalg import splu, svds
import numpy as np


def luq(matrix_a, tol):
    'This decompose a sparse matrix e using LUQ decomposition'

    # matrix_a = L [Ubar 0; 0 0] Q

    # The main reference is:
    # 1) Passivity Assessment and Model Order Reduction for Linear-Time-Invariant Descriptor Systems
    # in VLSI Circuit Simulation, Master Thesis by Zheng Zhang, 2010, section 3.9.3, page 93

    # matlab code is available at :
    # http://www.mathworks.com/matlabcentral/fileexchange/11120-null-space-of-a-sparse-matrix?focused=5073820&tab=function

    # USAGE: L, U, Q = luq(matrix_a, tol)
    # INPUT:
    #        matrix_a        a sparse matrix
    #        tol             uses to tolerance tol in separating zero and nonzero values
    # OUTPUT:
    #        L, U, Q         matrices

    # COMMENTS:
    #        based on lu decomposition

    if not issparse(matrix_a):
        raise ValueError('Matrix a is not a sparse matrix')

    pass


if __name__ == '__main__':

    matrix_a = random(10, 10, density=0.1, format='csc')

    print "\nmatrix_a = \n{}".format(matrix_a.todense())

    m, n = matrix_a.shape

    print "\nm = {}, n = {}".format(m, n)

    k = min(matrix_a.shape)

    print "\nk= {}".format(k)

    # test svd decomposition

    u, s, vt = svds(matrix_a, k - 1, tol=1.0e-12)
    matrix_s = np.diag(s)
    us = np.dot(u, matrix_s)
    usvt = np.dot(us, vt)

    print "\nu = \n{}, \ns = \n{}, \nvt = \n{}".format(u, s, vt)

    print "\nusvt = \n{}".format(usvt)

    print "\nusvt - matrix_a = {}".format(usvt - matrix_a)

    print "\nnorm of usvt - matrix_a = {}".format(np.linalg.norm(usvt - matrix_a))

    # test lu decomposition

    #lu = splu(matrix_a)
