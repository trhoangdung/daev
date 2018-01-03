'''
This module implements projectors used to decoupling the DAE equation
Dung Tran: Dec/2017
'''
from scipy.sparse import issparse
from scipy.linalg import svd
import time
import numpy as np
from daev.daes import index_1_daes, index_2_daes, index_3_daes


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
    'compute admissible projectors of regular matrix pencil (E, A) upto index 3'

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
    return_e_inv = None    # used to construct decoupled systems

    m = A0.shape[0]
    Im = np.eye(m, dtype=float)

    rank_E0 = np.linalg.matrix_rank(E0)
    if rank_E0 == m:
        print "\nsystem is index-0, dae can be convert to ode by inverse(E) * A"
    else:
        Q0, _ = orth_projector_on_ker_a(E0)
        E1 = E0 - np.dot(A0, Q0)
        rank_E1 = np.linalg.matrix_rank(E1)
        if rank_E1 == m:
            print "\nsystem is index-1"
            admissible_projectors.append(Q0)
            return_e_inv = np.linalg.inv(E1)
        else:
            Q1, _ = orth_projector_on_ker_a(E1)
            P0 = Im - Q0
            A1 = np.dot(A0, P0)
            E2 = E1 - np.dot(A1, Q1)
            rank_E2 = np.linalg.matrix_rank(E2)
            if rank_E2 == m:
                # print "\nsystem is index-2"
                # compute admissible Q1*
                E2_inv = np.linalg.inv(E2)
                E2_inv_A1 = np.dot(E2_inv, A1)
                admissible_Q1 = np.dot(-Q1, E2_inv_A1)
                admissible_projectors.append(Q0)
                admissible_projectors.append(admissible_Q1)
                return_e_inv = np.linalg.inv(E2)

            else:
                Q2, _ = orth_projector_on_ker_a(E2)
                P1 = Im - Q1
                A2 = np.dot(A1, P1)
                E3 = E2 - np.dot(A2, Q2)
                rank_E3 = np.linalg.matrix_rank(E3)
                if rank_E3 == m:
                    # print "\nsystem is index-3"
                    # compute admissible projectors Q2*, Q1*
                    E3_inv = np.linalg.inv(E3)
                    E3_inv_A2 = np.dot(E3_inv, A2)
                    admissible_Q2 = np.dot(-Q2, E3_inv_A2)
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
                    return_e_inv = np.linalg.inv(E3)
                else:
                    print "system has index > 3"
                    admissible_projectors = 'error'

    end = time.time()
    runtime = end - start
    return admissible_projectors, return_e_inv, runtime


def admissible_projectors_full(matrix_e, matrix_a):
    'constructing admissible projectors for regular matrix pencil (E, A) with arbitrary index'

    # return list of admissible projectors, length of the list = index of (E, A)

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
    e_mu_inv = None    # used to construct decoupled systems

    m = A0.shape[0]
    Im = np.eye(m, dtype=float)

    E = E0.copy()
    A = A0.copy()
    rank_E = np.linalg.matrix_rank(E)
    projectors = []    # contain projectors not yet admissible
    E_list = []
    A_list = []
    while (rank_E != m):
        Q, _ = orth_projector_on_ker_a(E)
        projectors.append(Q)    # used to construct admissible projectors
        E_list.append(E)    # used to construct admissible projectors
        A_list.append(A)
        E = E - np.dot(A, Q)
        A = np.dot(A, Im - Q)
        rank_E = np.linalg.matrix_rank(E)

    E_inv = np.linalg.inv(E)    # used to construct admissible projectors

    # constructing admissible projectors: "BACKWARD CONSTRUCTION ALGORITHM"
    ind = len(projectors)    # index of the system

    # the last admissible projector can be chosen by:
    # Q_ad_final = np.dot(-Q, np.dot(E_inv, A_list[ind - 1]))
    # or:
    Q_ad_final = Q
    if ind == 0:
        print "\nsystem has index-0, can be converted to ODE"
    elif ind == 1:
        admissible_projectors.append(projectors[0])
    elif ind == 2:
        admissible_projectors.append(projectors[0])
        admissible_projectors.append(Q_ad_final)
    elif ind > 2:
        # construct Vi = [Q_ad[ind-1], Q_ad[ind-2], ..., Q_ad[i + 1], E[i]]^T
        for i in xrange(0, ind - 1):
            admissible_projectors.append([])
        admissible_projectors.append(Q_ad_final)

        for i in xrange(ind - 2, -1, -1):
            Vi = E_list[i].copy()
            for j in xrange(i + 1, ind):
                Q_ad_j = admissible_projectors[j]
                Vi = np.vstack((Q_ad_j, Vi))
            print "\n %%%%%%%%%%%%%"
            print "\nshape of Vi = {}".format(Vi.shape)
            Q_ad_i, _ = orth_projector_on_ker_a(Vi)
            admissible_projectors[i] = Q_ad_i

    end = time.time()
    runtime = end - start

    return admissible_projectors, runtime


def test():
    'test methods'

    # E1, A1, _, _ = index_1_daes().RLC_circuit(1.0, 1.0, 1.0)
    # E2, A2, _, _ = index_2_daes().RL_network(1.0, 1.0)
    E3, A3, _, _ = index_3_daes().car_pendulum(1.0, 1.0, 1.0)

    # projs1, _ = admissible_projectors_full(E1, A1)
    # projs2, _ = admissible_projectors_full(E2, A2)
    projs3, _ = admissible_projectors_full(E3, A3)

    # print "admissible projectors 1 : \n{}".format(projs1)
    # print "admissible projectors 2 : \n{}".format(projs2)
    # print "\nnorm of Q1 * Q0 = {}".format(np.linalg.norm(np.dot(projs2[1], projs2[0])))
    print "admissible projectors 3 : \n{}".format(projs3)
    print "\nnorm of Q1 * Q0 = {}".format(np.linalg.norm(np.dot(projs3[1], projs3[0])))
    print "\nnorm of Q2 * Q0 = {}".format(np.linalg.norm(np.dot(projs3[2], projs3[0])))
    print "\nnorm of Q2 * Q1 = {}".format(np.linalg.norm(np.dot(projs3[2], projs3[1])))

if __name__ == '__main__':

    test()
