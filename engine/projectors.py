'''
This module implements projectors used to decompose the DAE equation
Dung Tran: Dec/2017
'''

from scipy.sparse import issparse, random
from scipy.linalg import svd
import time
import numpy as np


def null_space(matrix_a):
    'compute null space of a matrix_a using svd decomposition'

    start = time.time()
    print "\ncomputing null space ..."
    if issparse(matrix_a):
        a_mat = matrix_a.todense()
    else:
        assert isinstance(matrix_a, np.ndarray)
        a_mat = matrix_a

    m, n = a_mat.shape
    u_mat, s_vec, vt_mat = svd(a_mat)

    rank_a = np.linalg.matrix_rank(a_mat)
    v_mat = np.transpose(vt_mat)
    null_a = v_mat[:, rank_a:n]
    end = time.time()
    runtime = end - start
    print "\ncomputing null space is finished in {} seconds".format(runtime)

    return null_a, runtime


def orth_projector_on_ker_a(matrix_a):
    'implement orthogonal projector onto Ker of matrix a'

    # A*Q = 0, Q * Q = Q
    print "\ncomputing orthogonal projector on ker of matrix a ..."
    start = time.time()
    null_a, _ = null_space(matrix_a)
    projector = np.dot(null_a, np.transpose(null_a))
    end = time.time()
    runtime = end - start
    print "\ncomputing orthogonal projector is finished in {} seconds".format(runtime)

    return projector, runtime


def canonical_projector_on_ker_a(matrix_a):
    'implement canonical projector onto ker of matrix a'

    pass

if __name__ == '__main__':

    matrix_a = random(5, 5, density=0.1, format='csc')
    # dim = 1000, few seconds
    # dim = 2000, 20 seconds
    # dim = 4000, 182 seconds

    # accuracy is arround e-12

    null_a, _ = null_space(matrix_a)

    print "\nnull space of matrix _a is : \n{}".format(null_a)

    print "\nnorm of matrix_a * null_a = {}".format(np.linalg.norm(np.dot(matrix_a.todense(), null_a)))

    projector, _ = orth_projector_on_ker_a(matrix_a)

    print "\northogonal projector onto Ker(matrix_a) is : \n{}".format(projector)

    print "\nnorm of matrix_a * projector is : {}".format(np.linalg.norm(np.dot(matrix_a.todense(), projector)))
