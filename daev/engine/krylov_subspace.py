'''
This module implements krylov subspace method for solving linear ODE with very large matrix A.

This module utilizes the Krypy 2.1.7 Python package: INSTALLATION: sudo pip install krypy

The ERROR BOUND caused by Krylov subspace method is computed based the paper:

1) "ERROR BOUNDS FOR THE KRYLOV SUBSPACE METHODS FOR COMUPTATIONS OF MATRIX EXPONENTIALS", HAO WANG AND QIANG YE, SAM.J.MATRIX ANAL, 2017

Dung Tran: 4/16/2018
'''

from krypy.utils import arnoldi
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import expm
import numpy as np
import math
from daev.pdes import HeatOneDimension
from daev.daes import index_1_daes


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

        # Reference: ERROR BOUNDS FOR THE KRYLOV SUBSPACE METHODS FOR COMUPTATIONS OF MATRIX EXPONENTIALS
        # Theorem 3.1
        # Using Simpson's rule for integration

        assert isinstance(T, float) and T > 0, 'error: invalid time T'
        assert isinstance(
            num_steps, int) and num_steps > 0, 'error: invalid number of steps'

        assert self.matrix_v is not None, 'error: matrix V is None, run Arnoldi algorithm first'

        def compute_vA(self):
            'compute smallest eigenvalue of (A + A^T)/2'
            a_trans = self.matrix_a.transpose()
            new_a = (self.matrix_a + a_trans) / 2
            sm_eig, _ = eigs(new_a, k=1, which='SR')
            print "\nsm_eig = {}".format(sm_eig)
            return sm_eig.real

        def compute_gt(vA, t, T):
            'compute g(t) = exp((t - T)vA) at specific t'
            x = (t - T) * vA
            return math.exp(x)

        def compute_ht(Hk, ek, e1, t):
            'compute |h(t)|, h(t) = ek^T * e^(-t * Hk) * e1'
            Hk_t = expm(Hk.dot(-t))
            Hk_t_e1 = np.dot(Hk_t, e1)
            ht = np.dot(np.transpose(ek), Hk_t_e1)
            return abs(ht)

        # compute the left hand side of Theorem 3.1, the equation (3.6)
        # We are using the Simpson's rule for each step [0, step]:
        # https://en.wikipedia.org/wiki/Simpson%27s_rule

        def get_ht_list(self, T, num_steps):

            k = self.matrix_v.shape[1] - 1
            Ik = np.eye(k, dtype=float)
            ek = Ik[:, k - 1]
            e1 = Ik[:, 0]
            Hk = self.matrix_h[0: k, :]

            int_step = T / (2 * num_steps)
            ht_list = []
            for i in xrange(0, 2 * num_steps + 1):
                t = int_step * i
                ht_list.append(compute_ht(Hk, ek, e1, t))

            return ht_list

        def get_gt_list(self, T, num_steps):
            int_step = T / (2 * num_steps)
            vA = compute_vA(self)
            gt_list = []
            for i in xrange(0, 2 * num_steps + 1):
                gt_list.append(compute_gt(vA, int_step * i, T))

            return gt_list

        def compute_error_bound(self, T, num_steps):
            ht_list = get_ht_list(self, T, num_steps)
            gt_list = get_gt_list(self, T, num_steps)

            print "\nht_list = \n{}".format(ht_list)
            print "\ngt_list = \n{}".format(gt_list)

            zt = []
            for i in xrange(0, 2 * num_steps + 1):
                zt.append(ht_list[i] * gt_list[i])

            # compute integral
            int_res = 0
            step = T / num_steps

            for i in xrange(0, num_steps):
                j = 2 * i
                zt_total = zt[j] + 4 * zt[j + 1] + zt[j + 2]
                int_res = int_res + (step / 6) * zt_total

            return int_res

        int_res = compute_error_bound(self, T, num_steps)

        return int_res


def test():
    'test Krylov subspace method'

    heateq = HeatOneDimension(0.1, 1.0, 1.0, 1.0)
    _, matrix_a, _, _ = index_1_daes().RLC_circuit(1.0, 1.0, 1.0)
    # matrix_a, _ = heateq.get_odes(10)
    n = matrix_a.shape[0]
    vector_v = np.random.rand(n, 1)    # initial vector
    maxiter = 2                    # number of iteration
    Kry = KrylovSubspaceCPU()
    V, H = Kry.run_Arnoldi(matrix_a.tocsr(), vector_v, maxiter)

    print "\nmatrix A = \n{}".format(matrix_a.todense())
    print "\nvector v = \n{}".format(vector_v)

    print "\nmatrix V = \n{}".format(V)
    print "\nmatrix H = \n{}".format(H)
    print "\nshape of V = {}".format(V.shape)
    print "\nshape of H = {}".format(H.shape)

    int_res = Kry.get_error(1.0, 10)
    print "\nintegral result = {}".format(int_res)


if __name__ == '__main__':

    test()
