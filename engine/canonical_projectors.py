'''
This module implements canonical projectors and its related functions used to decompose the DAE equation
Dung Tran: Dec/2017
'''

from scipy.sparse import issparse, random
from scipy.linalg import svd
import time
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

    matrix_a = random(6, 6, density=0.1, format='csc')
    # dim = 1000, few seconds
    # dim = 2000, 20 seconds
    # dim = 4000, 182 seconds

    # accuracy is arround e-12

    print "\nmatrix_a = \n{}".format(matrix_a.todense())

    r = np.linalg.matrix_rank(matrix_a.todense())

    print "\nrank of matrix_a = {}".format(r)
    m, n = matrix_a.shape

    print "\nm = {}, n = {}".format(m, n)

    if m != n:
        raise ValueError('not a square matrix')

    # test svd decomposition
    start = time.time()
    u, s, vt = svd(matrix_a.todense())
    end = time.time()
    print "\nsvd computation time = {}".format(end - start)

    matrix_s = np.diag(s)
    us = np.dot(u, matrix_s)
    usvt = np.dot(us, vt)

    print "\nu = \n{}, \ns = \n{}, \nvt = \n{}".format(u, s, vt)

    print "\nusvt = \n{}".format(usvt)

    print "\nnorm of usvt - matrix_a = {}".format(np.linalg.norm(usvt - matrix_a))

    v = np.transpose(vt)
    null_a = v[:, r:n]

    print "\nnull space of matrix _a = spane of {}".format(null_a)
    if null_a != []:

        print "\nnorm of  matrix_a * null_1 = {}".format(np.linalg.norm(np.multiply(matrix_a.todense(), null_a[:, 0])))
        print "\nnorm of matrix_a * null_2 = {}".format(np.linalg.norm(np.multiply(matrix_a.todense(), null_a[:, 1])))

    else:
        print "\nmatrix_a is full rank"
