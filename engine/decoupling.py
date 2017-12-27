'''
This module implements decoupling techniques for high-index DAE
Dung Tran: Dec/2017
'''

import numpy as np


class DecoupledIndexOne(object):
    'Decoupled system of index-1 dae'

    # index-1 dae can be decoupled into a decoupled system:
    #     dot{x1} = A1 * x1 + B1 * u
    #     x2 = A2 * x1 + B2 * u
    #     y = C(x1 + x2)

    def __init__(self):
        self.name = 'DecoupledIndexOne'
        self.ode_matrix_a = None    # ode part
        self.ode_matrix_b = None    # ode part
        self.alg_matrix_a = None    # algebraic constraint part
        self.alg_matrix_b = None    # algebraic constraint part
        self.out_matrix_c = None    # output matrix

    def set_dynamics(self, ode_a_mat, ode_b_mat, alg_a_mat, alg_b_mat, c_mat):
        'set dynamics for decoupled system'

        assert isinstance(ode_a_mat, np.ndarray)
        assert isinstance(ode_b_mat, np.ndarray)

        assert isinstance(alg_a_mat, np.ndarray)
        assert isinstance(alg_b_mat, np.ndarray)
        assert isinstance(c_mat, np.ndarray)

        assert ode_a_mat.shape[0] == ode_a_mat.shape[1] == ode_b_mat.shape[0], 'error: inconsistency'
        assert alg_a_mat.shape[0] == alg_a_mat.shape[1] == alg_b_mat.shape[0], 'error: inconsistency'

        assert alg_a_mat.shape[0] == ode_a_mat.shape[0], 'error: inconsistency'
        assert c_mat.shape[1] == ode_a_mat.shape[0], 'error: inconsistency'

        self.ode_matrix_a = ode_a_mat
        self.ode_matrix_b = ode_b_mat
        self.alg_matrix_a = alg_a_mat
        self.alg_matrix_b = alg_b_mat
        self.out_matrix_c = c_mat


class DecoupledIndexTwo(object):
    'Decoupled system of index-2 dae'

    # index-2 dae can be decoupled into a system as follows:
    #    dot{x1} = A1 * x1 + B1 * u    (ode part)
    #    x2 = A2 * x1 + B2 * u         (algebraic constraint part 1)
    #    x3 = A3 * x1 + B3 * u + C3 * dot{x2}    (algebraic constraint part 2)
    #    y = c(x1 + x2 + x3)

    def __init__(self):
        self.name = 'DecoupledIndexTwo'
        self.ode_matrix_a = None    # ode part
        self.ode_matrix_b = None    # ode part
        self.alg1_matrix_a = None    # algebraic constraints part 1
        self.alg1_matrix_b = None    # algebraic constraints part 1
        self.alg2_matrix_a = None    # algebraic constraints part 2
        self.alg2_matrix_b = None    # algebraic constraints part 2
        self.alg2_matrix_c = None    # algebraic constraints part 2
        self.out_matrix_c = None     # output matrix

    def set_dynamics(self, ode_a_mat, ode_b_mat, alg1_a_mat,
                     alg1_b_mat, alg2_a_mat, alg2_b_mat, alg2_c_mat, c_mat):
        'set dynamics for decoupled system'

        assert isinstance(ode_a_mat, np.ndarray)
        assert isinstance(ode_b_mat, np.ndarray)

        assert isinstance(alg1_a_mat, np.ndarray)
        assert isinstance(alg1_b_mat, np.ndarray)
        assert isinstance(alg2_a_mat, np.ndarray)
        assert isinstance(alg2_b_mat, np.ndarray)
        assert isinstance(alg2_c_mat, np.ndarray)

        assert isinstance(c_mat, np.ndarray)

        assert ode_a_mat.shape[0] == ode_a_mat.shape[1] == ode_b_mat.shape[0], 'error: inconsistency'
        assert alg1_a_mat.shape[0] == alg1_a_mat.shape[1] == alg1_b_mat.shape[0], 'error: inconsistency'
        assert alg2_a_mat.shape[0] == alg2_a_mat.shape[1] == alg2_b_mat.shape[
            0] == alg2_c_mat.shape[0] == alg2_c_mat.shape[1], 'error: inconsistency'

        assert alg1_a_mat.shape[0] == ode_a_mat.shape[0] == alg2_a_mat.shape[0], 'error: inconsistency'
        assert c_mat.shape[1] == ode_a_mat.shape[0], 'error: inconsistency'

        self.ode_matrix_a = ode_a_mat
        self.ode_matrix_b = ode_b_mat
        self.alg1_matrix_a = alg1_a_mat
        self.alg1_matrix_b = alg1_b_mat
        self.alg2_matrix_a = alg2_a_mat
        self.alg2_matrix_b = alg2_b_mat
        self.alg2_matrix_c = alg2_c_mat
        self.out_matrix_c = c_mat


class DecoupledIndexThree(object):
    'Decoupled system of index-3 dae'

    # index-3 dae can be decoupled into a system as follows:
    #    ode:    dot{x1} = A1 * x1 + B1 * u
    #    alg1:    x2 = A2 * x1 + B2 * u
    #    alg2:    x3 = A3 * x1 + B3 * u + C3 * dot{x2}
    #    alg3:    x4 = A4 * x1 + B4 * u + C4 * dot{x3} + D4 * dot{x2}
    #    output:  y = C(x1 + x2 + x3 + x4)

    def __init__(self):
        self.name = 'DecoupledIndexThree'
        self.ode_matrix_a = None    # ode part
        self.ode_matrix_b = None    # ode part
        self.alg1_matrix_a = None    # alg1 part
        self.alg1_matrix_b = None    # alg1 part
        self.alg2_matrix_a = None    # alg2 part
        self.alg2_matrix_b = None    # alg2 part
        self.alg2_matrix_c = None    # alg2 part
        self.alg3_matrix_a = None    # alg3 part
        self.alg3_matrix_b = None    # alg3 part
        self.alg3_matrix_c = None    # alg3 part
        self.alg3_matrix_d = None    # alg3 part

    def set_dynamics(self, ode_a_mat, ode_b_mat, alg1_a_mat, alg1_b_mat, alg2_a_mat,
                     alg2_b_mat, alg2_c_mat, alg3_a_mat, alg3_b_mat, alg3_c_mat, alg3_d_mat, c_mat):
        'set dynamics for decoupled system'

        assert isinstance(ode_a_mat, np.ndarray)
        assert isinstance(ode_b_mat, np.ndarray)

        assert isinstance(alg1_a_mat, np.ndarray)
        assert isinstance(alg1_b_mat, np.ndarray)
        assert isinstance(alg2_a_mat, np.ndarray)
        assert isinstance(alg2_b_mat, np.ndarray)
        assert isinstance(alg2_c_mat, np.ndarray)
        assert isinstance(alg3_a_mat, np.ndarray)
        assert isinstance(alg3_b_mat, np.ndarray)
        assert isinstance(alg3_c_mat, np.ndarray)
        assert isinstance(alg3_d_mat, np.ndarray)
        assert isinstance(c_mat, np.ndarray)

        assert ode_a_mat.shape[0] == ode_a_mat.shape[1] == ode_b_mat.shape[0], 'error: inconsistency'
        assert alg1_a_mat.shape[0] == alg1_a_mat.shape[1] == alg1_b_mat.shape[0], 'error: inconsistency'
        assert alg2_a_mat.shape[0] == alg2_a_mat.shape[1] == alg2_b_mat.shape[
            0] == alg2_c_mat.shape[0] == alg2_c_mat.shape[1], 'error: inconsistency'

        assert alg3_a_mat.shape[0] == alg3_a_mat.shape[1] == alg3_b_mat.shape[
            0] == alg3_c_mat.shape[0] == alg3_c_mat.shape[
                1] == alg3_d_mat.shape[0] == alg3_d_mat.shape[1], 'error: inconsistency'

        assert alg1_a_mat.shape[0] == ode_a_mat.shape[0] == alg2_a_mat.shape[0] == alg3_a_mat.shape[0], 'error: inconsistency'
        assert c_mat.shape[1] == ode_a_mat.shape[0], 'error: inconsistency'

        self.ode_matrix_a = ode_a_mat
        self.ode_matrix_b = ode_b_mat
        self.alg1_matrix_a = alg1_a_mat
        self.alg1_matrix_b = alg1_b_mat
        self.alg2_matrix_a = alg2_a_mat
        self.alg2_matrix_b = alg2_b_mat
        self.alg2_matrix_c = alg2_c_mat
        self.alg3_matrix_a = alg3_a_mat
        self.alg3_matrix_b = alg3_b_mat
        self.alg3_matrix_c = alg3_c_mat
        self.alg3_matrix_d = alg3_d_mat
        self.out_matrix_c = c_mat
