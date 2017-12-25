'''
This benchmark tests construction of addmissible projector iteratively
Dung Tran: Dec/2017
'''

from daes import index_2_daes
from projectors import orth_projector_on_ker_a
import numpy as np


if __name__ == '__main__':

    E, A, _, _ = index_2_daes().two_interconnected_rotating_masses(1.0, 1.0)

    print "\nE = \n{}, A = \n{}".format(E, A)

    E0 = E
    A0 = A
    Q0, _ = orth_projector_on_ker_a(E0)

    print "\nQ0 = \n{}".format(Q0)

    m, _ = Q0.shape

    P0 = np.eye(m, dtype=float) - Q0

    E1 = E0 + np.dot(A0, Q0)
    A1 = np.dot(A0, P0)

    print "\nE1 = \n{}, \nA1 = \n{}".format(E1, A1)

    Q1, _ = orth_projector_on_ker_a(E1)

    print "\nQ1 = \n{}".format(Q1)

    print "\nQ1 * Q0 = \n{}".format(np.dot(Q1, Q0))

    P1 = np.eye(m, dtype=float) - Q1

    E2 = E1 + np.dot(A1, Q1)
    A2 = np.dot(A1, P1)

    print "\nE2 = \n{}, \nA2 = \n{}".format(E2, A2)
