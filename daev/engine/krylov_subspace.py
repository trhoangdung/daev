'''
This module implements krylov subspace method for solving linear ODE with very large matrix A.

This module utilizes the Krypy 2.1.7 Python package: INSTALLATION: sudo pip install krypy

The ERROR BOUND caused by Krylov subspace method is computed based the paper:

1) "ERROR BOUNDS FOR THE KRYLOV SUBSPACE METHODS FOR COMUPTATIONS OF MATRIX EXPONENTIALS", HAO WANG AND QIANG YE, SAM.J.MATRIX ANAL, 2017

Dung Tran: 4/16/2018
'''

from krypy.utils import arnoldi
from scipy.sparse import csr_matrix
from scipy.linalg import expm
import numpy as np
from daev.pdes import HeatOneDimension


class KrylovSubspaceCPU(object):
    'Implement Krylov subspace method using CPU'

    def __init__(self):
        self.matrix_a = None
        self.matrix_v = None
        self.vector_v = None
        self.matrix_h = None
        self.error = None

    def run_Arnoldi(self, matrix_a, vector_v, maxiter):
        'compute matrix V and H, A * V_n = V_n * H_n'

        assert isinstance(
            matrix_a, csr_matrix) and matrix_a.shape[0] == matrix_a.shape[1], 'error: matrix_a is not a csr_matrix or not a square matrix'
        assert isinstance(
            vector_v, np.ndarray) and vector_v.shape[1] == 1, 'error: vector_v is not a numpy vector or has > 1 column'
        assert vector_v.shape[0] == matrix_a.shape[0], 'error: inconsistency between vector v and matrix_a'
        assert isinstance(
            maxiter, int) and maxiter >= 2, 'error: invalid numer of iteration'
        self.matrix_a = matrix_a
        self.vector_v = vector_v
        self.matrix_v, self.matrix_h = arnoldi(
            matrix_a, vector_v, maxiter=maxiter)

        return self.matrix_v, self.matrix_h

    def get_error(self, T, num_steps):
        'get error bound, computation is based on Theorem 3.1 of paper 1)'

        # Using Simpson's rule for integration of matrix expoinential

        assert isinstance(T, float) and T > 0, 'error: invalid time T'
        assert isinstance(
            num_steps, int) and num_steps > 0, 'error: invalid number of steps'

        assert self.matrix_v is not None, 'error: matrix V is None, run Arnoldi algorithm first'
        n = self.matrix_v.shape[0]
        k = self.matrix_v.shape[1] - 1
        Ik = np.eye(k, dtype=float)
        ek = Ik[:, k - 1]
        e1 = Ik[:, 0]
        Hk = self.matrix_h[0: k, :]
        step = T / num_steps
        ht_list = []

        def compute_ht(self, t):
            Hk_t = expm(Hk.dot(-t))
            Hk_t_e1 = np.dot(Hk_t, e1)
            ht = np.dot(np.transpose(ek), Hk_t_e1)
            return ht

        for i in xrange(0, num_steps + 1):
            ht = compute_ht(self, step * i)
            ht_list.append(ht)

        return ht_list


def test():
    'test Krylov subspace method'

    heateq = HeatOneDimension(0.1, 1.0, 1.0, 1.0)

    matrix_a, _ = heateq.get_odes(10)
    n = matrix_a.shape[0]
    vector_v = np.random.rand(n, 1)    # initial vector
    maxiter = 4                    # number of iteration
    Kry = KrylovSubspaceCPU()
    V, H = Kry.run_Arnoldi(matrix_a.tocsr(), vector_v, maxiter)

    print "\nmatrix A = \n{}".format(matrix_a.todense())
    print "\nvector v = \n{}".format(vector_v)

    print "\nmatrix V = \n{}".format(V)
    print "\nmatrix H = \n{}".format(H)
    print "\nshape of V = {}".format(V.shape)
    print "\nshape of H = {}".format(H.shape)

    ht_list = Kry.get_error(1.0, 10)
    print "\nht list = {}".format(ht_list)

if __name__ == '__main__':

    test()
