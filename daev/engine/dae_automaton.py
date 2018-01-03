'''
This module implements an automaton with dae dynamics
Dung Tran: Dec/2017
'''

from scipy.sparse import csc_matrix
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


class AutonomousDaeAutomation(object):
    'implement automaton with DAE dynamics'

    # DAE automaton has the form of : Ex' = Ax, y = Cx, E is singular
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
