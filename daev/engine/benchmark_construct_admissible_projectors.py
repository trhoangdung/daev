'''
This benchmark tests construction of addmissible projector iteratively
Dung Tran: Dec/2017
'''

from daes import index_2_daes, index_3_daes
from projectors import orth_projector_on_ker_a
import numpy as np


def test_index2():
    'test for index 2 examples'

    E, A, _, _ = index_2_daes().two_interconnected_rotating_masses(1.0, 1.0)

    print "\nE = \n{}, A = \n{}".format(E, A)

    E0 = E
    A0 = A
    Q0, _ = orth_projector_on_ker_a(E0)

    print "\nQ0 = \n{}".format(Q0)

    m, _ = Q0.shape
    Im = np.eye(m, dtype=float)
    P0 = Im - Q0
    E1 = E0 + np.dot(A0, Q0)
    A1 = np.dot(A0, P0)

    Q1, _ = orth_projector_on_ker_a(E1)

    print "\nQ1 = \n{}".format(Q1)
    print "\nQ1 * Q0 = \n{}".format(np.dot(Q1, Q0))

    E2 = E1 + np.dot(A1, Q1)
    rank_E2 = np.linalg.matrix_rank(E2)
    if rank_E2 == m:
        print "\nE2 is nonsingluar"
    else:
        print "\nE2 is singluar"

    if rank_E2 == m:
        # compute admissible projectors
        E2_inv = np.linalg.inv(E2)
        E2_inv_A1 = np.dot(E2_inv, A1)
        admissible_Q1 = np.dot(Q1, E2_inv_A1)
        print "\nadmissible Q1 = \n{}".format(admissible_Q1)
        admissible_Q0 = Q0
        print "\nadmissible Q0 = \n{}".format(admissible_Q0)

        print "\nnorm of admissible Q1 = {}".format(np.linalg.norm(admissible_Q1))
        print "\nnorm of admissible Q0 = {}".format(np.linalg.norm(admissible_Q0))
        print "\nnorm of admissible Q1 * admissible Q0 = {}".format(np.linalg.norm(np.dot(admissible_Q1, admissible_Q0)))


def test_index3():
    'test for index 3 examples'

    E, A, _, _ = index_3_daes().car_pendulum(1.0, 1.0, 1.0)
    print "\nE = \n{}, \nA = \n{}".format(E, A)

    E0 = E
    A0 = A
    Q0, _ = orth_projector_on_ker_a(E0)
    print "\nQ0 = \n{}".format(Q0)
    m, _ = Q0.shape
    Im = np.eye(m, dtype=float)
    P0 = Im - Q0

    E1 = E0 + np.dot(A0, Q0)
    A1 = np.dot(A0, P0)
    Q1, _ = orth_projector_on_ker_a(E1)
    print "\nQ1 = \n{}".format(Q1)

    E2 = E1 + np.dot(A1, Q1)
    P1 = Im - Q1
    A2 = np.dot(A1, P1)
    Q2, _ = orth_projector_on_ker_a(E2)
    print"\nQ2 = \n{}".format(Q2)

    E3 = E2 + np.dot(A2, Q2)
    rank_E3 = np.linalg.matrix_rank(E3)
    if rank_E3 == m:
        print "\nE3 is nonsingular"
    else:
        print "\nE3 is singular"

    if rank_E3 == m:
        E3_inv = np.linalg.inv(E3)
        E3_inv_A2 = np.dot(E3_inv, A2)
        admissible_Q2 = np.dot(Q2, E3_inv_A2)
        admissible_P2 = Im - admissible_Q2
        E3_inv_A1 = np.dot(E3_inv, A1)
        ad_P2_E3_inv_A1 = np.dot(admissible_P2, E3_inv_A1)
        admissible_Q1 = np.dot(Q1, ad_P2_E3_inv_A1)
        print "\nadmissible Q2 = \n{}".format(admissible_Q2)
        print "\nadmissible Q1 = \n {}".format(admissible_Q1)

        # compute admissible Q0
        V = np.vstack((admissible_Q2, admissible_Q1, E0))
        admissible_Q0, _ = orth_projector_on_ker_a(V)
        print "\nadmissible Q0 = \n{}".format(admissible_Q0)

        print "\nnorm of admissible Q2 =  {}".format(np.linalg.norm(admissible_Q2))
        print "\nnorm of admissible Q1 =  {}".format(np.linalg.norm(admissible_Q1))
        print "\nnorm of admissible Q0 =  {}".format(np.linalg.norm(admissible_Q0))

        print "\nnorm of admissible Q2 * admissible Q1 = {}".format(np.linalg.norm(np.dot(admissible_Q2, admissible_Q1)))
        print "\nnorm of admissible Q2 * admissible Q0 = {}".format(np.linalg.norm(np.dot(admissible_Q2, admissible_Q0)))
        print "\nnorm of admissible Q1 * admissible Q0 = {}".format(np.linalg.norm(np.dot(admissible_Q1, admissible_Q0)))
        print "\nnorm of E0 * admissible Q0 = {}".format(np.linalg.norm(np.dot(E0, admissible_Q0)))


if __name__ == '__main__':

    #test_index2()
    test_index3()
