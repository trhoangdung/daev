'''
This module implements an automaton with dae dynamics
Dung Tran: Dec/2017
'''

from scipy.sparse import csc_matrix, eye, vstack, hstack
from scipy.linalg import inv
import numpy as np


class DaeAutomation(object):
    'implement automaton with DAE dynamics'

    # DAE automaton has the form of : Ex' = Ax + Bu, y = Cx, E is singular
    def __init__(self):
        self.matrix_e = None
        self.matrix_a = None
        self.matrix_b = None
        self.matrix_c = None

    def set_dynamics(self, matrix_e, matrix_a, matrix_b, matrix_c):
        'set dynamcis for dae automaton'

        assert isinstance(matrix_e, csc_matrix)
        assert isinstance(matrix_a, csc_matrix)
        assert isinstance(matrix_b, csc_matrix)
        assert isinstance(matrix_c, csc_matrix)

        assert matrix_e.shape == matrix_a.shape, 'inconsistent matrices'
        assert matrix_a.shape[0] == matrix_a.shape[1] == matrix_b.shape[0], 'inconsistent matrices'
        assert matrix_c.shape[1] == matrix_a.shape[0], 'inconsistent matrices'
        n = matrix_e.shape[0]
        rank_e = np.rank(matrix_e.todense())

        if rank_e == n:
            print "matrix_e is non-singluar, dae is equivalent to ode"
            inv_e = inv(matrix_e)
            self.matrix_a = inv_e * matrix_a
            self.matrix_b = inv_e * matrix_b
            self.matrix_c = self.matrix_c
        elif rank_e < n:
            self.matrix_e = matrix_e
            self.matrix_a = matrix_a
            self.matrix_b = matrix_b
            self.matrix_c = matrix_c

    def convert_to_autonomous_dae(self, u_mat):
        'convert dae to an autonomous dae'

        # if the input u satisfy: u' = Gu, Ex' = Ax + Bu, y = Cx can be converted into
        # [E 0; 0 I] * [x' u']^T = [A B; 0 G][x u]^T, y = [C 0]* [x u]^T

        # if G == 0, we have an affine DAE

        assert isinstance(u_mat, csc_matrix)
        assert u_mat.shape[0] == u_mat.shape[1] == self.matrix_b.shape[1], 'error: inconsistent u_mat and self.matrix_b'

        m = u_mat.shape[0]
        n = self.matrix_b.shape[0]
        Im = eye(m)
        Zm1 = csc_matrix((n, m), dtype=float)
        Zm2 = csc_matrix((m, n), dtype=float)
        E1 = hstack((self.matrix_e, Zm1), format='csc')
        E2 = hstack((Zm2, Im), format='csc')
        new_E = vstack((E1, E2), format='csc')
        A1 = hstack((self.matrix_a, self.matrix_b), format='csc')
        A2 = hstack((Zm2, u_mat), format='csc')
        new_A = vstack((A1, A2), format='csc')

        nC = self.matrix_c.shape[0]
        Zm3 = csc_matrix((nC, m), dtype=float)
        new_C = hstack((self.matrix_c, Zm3), format='csc')
        autonomous_dae = AutonomousDaeAutomation()
        autonomous_dae.set_dynamics(new_E, new_A, new_C)

        return autonomous_dae


class AutonomousDaeAutomation(object):
    'implement autonomous automaton with DAE dynamics'

    # autonomous DAE automaton has the form of : Ex' = Ax, y = Cx, E is singular
    def __init__(self):
        self.matrix_e = None
        self.matrix_a = None
        self.matrix_c = None

    def set_dynamics(self, matrix_e, matrix_a, matrix_c):
        'set dynamcis for dae automaton'

        assert isinstance(matrix_e, csc_matrix)
        assert isinstance(matrix_a, csc_matrix)
        assert isinstance(matrix_c, csc_matrix)

        assert matrix_e.shape == matrix_a.shape, 'inconsistent matrices'
        assert matrix_a.shape[0] == matrix_a.shape[1], 'invalid matrix_a'
        assert matrix_c.shape[1] == matrix_a.shape[0], 'inconsistent matrices'
        n = matrix_e.shape[0]
        rank_e = np.rank(matrix_e.todense())

        if rank_e == n:
            print "matrix_e is non-singluar, dae is equivalent to ode"
            inv_e = inv(matrix_e)
            self.matrix_a = inv_e * matrix_a
            self.matrix_c = self.matrix_c
        elif rank_e < n:
            self.matrix_e = matrix_e
            self.matrix_a = matrix_a
            self.matrix_c = matrix_c
