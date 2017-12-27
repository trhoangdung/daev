'''
This module implements projectors used to decoupling the DAE equation
Dung Tran: Dec/2017
'''
from scipy.sparse import issparse
from scipy.linalg import svd
import time
import numpy as np
from daes import index_2_daes, index_3_daes


def null_space(matrix_a):
    'compute null space of a matrix_a using svd decomposition'

    start = time.time()
    # print "\ncomputing null space ..."
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
    # print "\ncomputing null space is finished in {} seconds".format(runtime)

    return null_a, runtime


def orth_projector_on_ker_a(matrix_a):
    'implement orthogonal projector onto Ker of matrix a'

    # A*Q = 0, Q * Q = Q
    # print "\ncomputing orthogonal projector on ker of matrix a ..."
    start = time.time()
    null_a, _ = null_space(matrix_a)
    projector = np.dot(null_a, np.transpose(null_a))
    end = time.time()
    runtime = end - start
    # print "\ncomputing orthogonal projector is finished in {} seconds".format(runtime)

    return projector, runtime


def admissible_projectors(matrix_e, matrix_a):
    'compute admissible projectors of regular matrix pencil (E, A)'

    # references:
    #    1) An efficient projector-based passivity test for descriptor systems
    #    2) Cannonical projectors for linear differential algebraic equations

    #    we can compute admissible projector for system with index upto 3
    #    1) limits to index 2

    # return list of admissible projectors, length of the list is the system index

    start = time.time()

    if issparse(matrix_e):
        E0 = matrix_e.todense()
        assert E0.shape[0] == E0.shape[1], 'invalid matrix E'
    else:
        assert isinstance(matrix_e, np.ndarray)
        E0 = matrix_e
        assert E0.shape[0] == E0.shape[1], 'invalid matrix E'

    if issparse(matrix_a):
        A0 = matrix_e.todense()
        assert A0.shape[0] == A0.shape[1], 'invalid matrix A'
    else:
        assert isinstance(matrix_a, np.ndarray)
        A0 = matrix_a
        assert A0.shape[0] == A0.shape[1], 'invalid matrix A'

    assert A0.shape[0] == E0.shape[0], 'inconsistent matrices'

    admissible_projectors = []

    m = A0.shape[0]

    Im = np.eye(m, dtype=float)

    rank_E0 = np.linalg.matrix_rank(E0)
    if rank_E0 == m:
        print "\nsystem is index-0, dae can be convert to ode by inverse(E) * A"
    else:
        Q0, _ = orth_projector_on_ker_a(E0)
        E1 = E0 + np.dot(A0, Q0)
        rank_E1 = np.linalg.matrix_rank(E1)
        if rank_E1 == m:
            print "\nsystem is index-1"
            admissible_projectors.append(Q0)
        else:
            Q1, _ = orth_projector_on_ker_a(E1)
            P0 = Im - Q0
            A1 = np.dot(A0, P0)
            E2 = E1 + np.dot(A1, Q1)
            rank_E2 = np.linalg.matrix_rank(E2)
            if rank_E2 == m:
                # print "\nsystem is index-2"
                # compute admissible Q1*
                E2_inv = np.linalg.inv(E2)
                E2_inv_A1 = np.dot(E2_inv, A1)
                admissible_Q1 = np.dot(Q1, E2_inv_A1)
                admissible_projectors.append(Q0)
                admissible_projectors.append(admissible_Q1)

            else:
                Q2, _ = orth_projector_on_ker_a(E2)
                P1 = Im - Q1
                A2 = np.dot(A1, P1)
                E3 = E2 + np.dot(A2, Q2)
                rank_E3 = np.linalg.matrix_rank(E3)
                if rank_E3 == m:
                    # print "\nsystem is index-3"
                    # compute admissible projectors Q2*, Q1*
                    E3_inv = np.linalg.inv(E3)
                    E3_inv_A2 = np.dot(E3_inv, A2)
                    admissible_Q2 = np.dot(Q2, E3_inv_A2)
                    admissible_P2 = Im - admissible_Q2
                    E3_inv_A1 = np.dot(E3_inv, A1)
                    ad_P2_E3_inv_A1 = np.dot(admissible_P2, E3_inv_A1)
                    admissible_Q1 = np.dot(Q1, ad_P2_E3_inv_A1)
                    # compute admissible projector Q0*
                    V = np.vstack((admissible_Q2, admissible_Q1, E0))
                    admissible_Q0, _ = orth_projector_on_ker_a(V)

                    admissible_projectors.append(admissible_Q0)
                    admissible_projectors.append(admissible_Q1)
                    admissible_projectors.append(admissible_Q2)

                else:
                    print "system has index > 3"
                    admissible_projectors = 'error'

    end = time.time()
    runtime = end - start
    return admissible_projectors, runtime


if __name__ == '__main__':

    E1, A1, _, _ = index_2_daes().RL_network(1.0, 1.0)
    adm_proj_1, rt1 = admissible_projectors(E1, A1)
    print "\nlist of projectors : = {}".format(adm_proj_1)
    print "\nruntime = {}".format(rt1)

    E2, A2, _, _ = index_3_daes().car_pendulum(1.0, 1.0, 1.0)
    adm_proj_2, rt2 = admissible_projectors(E2, A2)
    print "\nlist of projectors : = {}".format(adm_proj_2)
    print "\nruntime = {}".format(rt2)
