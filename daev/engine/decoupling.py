'''
This module implements decoupling techniques for high-index DAE
Dung Tran: Dec/2017
'''

import numpy as np
from daev.engine.dae_automaton import DaeAutomation, AutonomousDaeAutomation
from daev.engine.projectors import admissible_projectors


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


class AutonomousDecoupledIndexOne(object):
    'AutonomousDecoupled system of index-1 dae'

    # autonomous index-1 dae can be decoupled into a decoupled system:
    #     dot{x1} = A1 * x1
    #     x2 = A2 * x1
    #     y = C(x1 + x2)

    def __init__(self):
        self.name = 'AutonomousDecoupledIndexOne'
        self.ode_matrix_a = None    # ode part
        self.alg_matrix_a = None    # algebraic constraint part
        self.out_matrix_c = None    # output matrix

    def set_dynamics(self, ode_a_mat, alg_a_mat, c_mat):
        'set dynamics for decoupled system'

        assert isinstance(ode_a_mat, np.ndarray)
        assert isinstance(alg_a_mat, np.ndarray)
        assert isinstance(c_mat, np.ndarray)

        assert ode_a_mat.shape[0] == ode_a_mat.shape[1], 'error: invalid ode_a_mat'
        assert alg_a_mat.shape[0] == alg_a_mat.shape[1], 'error: invalid alg_a_mat'

        assert alg_a_mat.shape[0] == ode_a_mat.shape[0], 'error: inconsistency'
        assert c_mat.shape[1] == ode_a_mat.shape[0], 'error: inconsistency'

        self.ode_matrix_a = ode_a_mat
        self.alg_matrix_a = alg_a_mat
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


class AutonomousDecoupledIndexTwo(object):
    'AutonomousDecoupled system of index-2 dae'

    # autonomous index-2 dae can be decoupled into a system as follows:
    #    dot{x1} = A1 * x1     (ode part)
    #    x2 = A2 * x1          (algebraic constraint part 1)
    #    x3 = A3 * x1 + C3 * dot{x2}    (algebraic constraint part 2)
    #    y = c(x1 + x2 + x3)

    def __init__(self):
        self.name = 'AutonomousDecoupledIndexTwo'
        self.ode_matrix_a = None    # ode part
        self.alg1_matrix_a = None    # algebraic constraints part 1
        self.alg2_matrix_a = None    # algebraic constraints part 2
        self.alg2_matrix_c = None    # algebraic constraints part 2
        self.out_matrix_c = None     # output matrix

    def set_dynamics(self, ode_a_mat, alg1_a_mat, alg2_a_mat, alg2_c_mat, c_mat):
        'set dynamics for decoupled system'

        assert isinstance(ode_a_mat, np.ndarray)
        assert isinstance(alg1_a_mat, np.ndarray)
        assert isinstance(alg2_a_mat, np.ndarray)
        assert isinstance(alg2_c_mat, np.ndarray)

        assert isinstance(c_mat, np.ndarray)

        assert ode_a_mat.shape[0] == ode_a_mat.shape[1], 'error: invalid ode_a_mat'
        assert alg1_a_mat.shape[0] == alg1_a_mat.shape[1], 'error: invalid alg1_a_mat'
        assert alg2_a_mat.shape[0] == alg2_a_mat.shape[1] == alg2_c_mat.shape[0] == alg2_c_mat.shape[1], 'error: inconsistency'

        assert alg1_a_mat.shape[0] == ode_a_mat.shape[0] == alg2_a_mat.shape[0], 'error: inconsistency'
        assert c_mat.shape[1] == ode_a_mat.shape[0], 'error: inconsistency'

        self.ode_matrix_a = ode_a_mat
        self.alg1_matrix_a = alg1_a_mat
        self.alg2_matrix_a = alg2_a_mat
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
        self.out_matrix_c = None    # output matrix

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


class AutonomousDecoupledIndexThree(object):
    'Autonomous Decoupled system of index-3 dae'

    # autonomous index-3 dae can be decoupled into a system as follows:
    #    ode:    dot{x1} = A1 * x1
    #    alg1:    x2 = A2 * x1
    #    alg2:    x3 = A3 * x1 + C3 * dot{x2}
    #    alg3:    x4 = A4 * x1 + C4 * dot{x3} + D4 * dot{x2}
    #    output:  y = C(x1 + x2 + x3 + x4)

    def __init__(self):
        self.name = 'AutonomousDecoupledIndexThree'
        self.ode_matrix_a = None    # ode part
        self.alg1_matrix_a = None    # alg1 part
        self.alg2_matrix_a = None    # alg2 part
        self.alg2_matrix_c = None    # alg2 part
        self.alg3_matrix_a = None    # alg3 part
        self.alg3_matrix_c = None    # alg3 part
        self.alg3_matrix_d = None    # alg3 part
        self.out_matrix_c = None    # output matrix

    def set_dynamics(self, ode_a_mat, alg1_a_mat, alg2_a_mat,
                     alg2_c_mat, alg3_a_mat, alg3_c_mat, alg3_d_mat, c_mat):
        'set dynamics for decoupled system'

        assert isinstance(ode_a_mat, np.ndarray)
        assert isinstance(alg1_a_mat, np.ndarray)
        assert isinstance(alg2_a_mat, np.ndarray)
        assert isinstance(alg2_c_mat, np.ndarray)
        assert isinstance(alg3_a_mat, np.ndarray)
        assert isinstance(alg3_c_mat, np.ndarray)
        assert isinstance(alg3_d_mat, np.ndarray)
        assert isinstance(c_mat, np.ndarray)

        assert ode_a_mat.shape[0] == ode_a_mat.shape[1], 'error: invalid ode_a_mat'
        assert alg1_a_mat.shape[0] == alg1_a_mat.shape[1], 'error: invalid alg1_a_mat'
        assert alg2_a_mat.shape[0] == alg2_a_mat.shape[1] == alg2_c_mat.shape[0] == alg2_c_mat.shape[1], 'error: inconsistency'

        assert alg3_a_mat.shape[0] == alg3_a_mat.shape[1] == alg3_c_mat.shape[0] == alg3_c_mat.shape[
            1] == alg3_d_mat.shape[0] == alg3_d_mat.shape[1], 'error: inconsistency'

        assert alg1_a_mat.shape[0] == ode_a_mat.shape[0] == alg2_a_mat.shape[0] == alg3_a_mat.shape[0], 'error: inconsistency'
        assert c_mat.shape[1] == ode_a_mat.shape[0], 'error: inconsistency'

        self.ode_matrix_a = ode_a_mat

        self.alg1_matrix_a = alg1_a_mat
        self.alg2_matrix_a = alg2_a_mat
        self.alg2_matrix_c = alg2_c_mat
        self.alg3_matrix_a = alg3_a_mat
        self.alg3_matrix_c = alg3_c_mat
        self.alg3_matrix_d = alg3_d_mat
        self.out_matrix_c = c_mat


class Decoupling(object):
    'implement decoupling techniques using admissible projectors'

    def __init__(self):

        self.decoupled_sys = None    # return decoupled system
        self.status = None    # return status of decoupling process

    def get_decoupled_system(self, dae_automaton):
        'get decoupled system from an dae automaton'

        assert isinstance(dae_automaton, DaeAutomation)

        matrix_e = dae_automaton.matrix_e
        matrix_a = dae_automaton.matrix_a
        matrix_b = dae_automaton.matrix_b
        matrix_c = dae_automaton.matrix_c

        n, _ = matrix_a.shape[0]
        In = np.eye(n, dtype=float)

        adm_projs, e_mu_inv, _ = admissible_projectors(matrix_e, matrix_a)

        if adm_projs == 'error':
            print "\nerror: dae system has index larger than 3"
            self.status = 'error'
        else:
            assert isinstance(adm_projs, list), 'error: not a list of admissible projectors'
            ind = len(adm_projs)

            if ind == 1:

                decoupled_sys = DecoupledIndexOne()
                Q0 = adm_projs[0]
                P0 = In - Q0
                A0 = matrix_a
                e_mu_inv_A0 = np.dot(e_mu_inv, A0)

                # ode part
                ode_a_mat = np.dot(P0, e_mu_inv_A0)
                e_mu_inv_B = np.dot(e_mu_inv, matrix_b)
                ode_b_mat = np.dot(P0, e_mu_inv_B)

                # alg part
                alg_a_mat = np.dot(Q0, e_mu_inv_A0)
                alg_b_mat = np.dot(Q0, e_mu_inv_B)
                decoupled_sys.set_dynamics(ode_a_mat, ode_b_mat, alg_a_mat, alg_b_mat, matrix_c)

                self.decoupled_sys = decoupled_sys
                self.status = 'success'

            elif ind == 2:

                decoupled_sys = DecoupledIndexTwo()
                Q0 = adm_projs[0]
                Q1 = adm_projs[1]
                P0 = In - Q0
                P1 = In - Q1
                # ode part
                P0_P1 = np.dot(P0, P1)
                A2 = np.dot(matrix_a, P0_P1)
                E2_inv_A2 = np.dot(e_mu_inv, A2)
                E2_inv_B = np.dot(e_mu_inv, matrix_b)
                ode_a_mat = np.dot(P0_P1, E2_inv_A2)
                ode_b_mat = np.dot(P0_P1, E2_inv_B)

                # alg1 part
                alg1_a_mat = np.dot(Q1, E2_inv_A2)
                alg1_b_mat = np.dot(Q1, E2_inv_B)

                # alg2 part
                Q0_P1 = np.dot(Q0, P1)
                alg2_a_mat = np.dot(Q0_P1, E2_inv_A2)
                alg2_b_mat = np.dot(Q0_P1, E2_inv_B)
                alg2_c_mat = np.dot(Q0, Q1)

                decoupled_sys.set_dynamics(ode_a_mat, ode_b_mat, alg1_a_mat, alg1_b_mat, alg2_a_mat, \
                                               alg2_b_mat, alg2_c_mat, matrix_c)

                self.decoupled_sys = decoupled_sys
                self.status = 'success'

            elif ind == 3:

                decoupled_sys = DecoupledIndexThree()
                Q0 = adm_projs[0]
                Q1 = adm_projs[1]
                Q2 = adm_projs[2]
                P0 = In - Q0
                P1 = In - Q1
                P2 = In - Q2

                P0_P1_P2 = np.dot(P0, np.dot(P1, P2))
                A3 = np.dot(A0, P0_P1_P2)
                E3_inv_A3 = np.dot(e_mu_inv, A3)
                E3_inv_B = np.dot(e_mu_inv, matrix_b)
                # ode part
                ode_a_mat = np.dot(P0_P1_P2, E3_inv_A3)
                ode_b_mat = np.dot(P0_P1_P2, E3_inv_B)

                # alg1 part
                P0_P1_Q2 = np.dot(P0, np.dot(P1, Q2))
                alg1_a_mat = np.dot(P0_P1_Q2, E3_inv_A3)
                alg1_b_mat = np.dot(P0_P1_Q2, E3_inv_B)

                # alg2 part
                P0_Q1_P2 = np.dot(P0, np.dot(Q1, P2))
                alg2_a_mat = np.dot(P0_Q1_P2, E3_inv_A3)
                alg2_b_mat = np.dot(P0_Q1_P2, E3_inv_B)
                alg2_c_mat = np.dot(P0, np.dot(Q1, Q2))

                # alg3 part
                Q0_P1_P2 = np.dot(Q0, np.dot(P1, P2))
                alg3_a_mat = np.dot(Q0_P1_P2, E3_inv_A3)
                alg3_b_mat = np.dot(Q0_P1_P2, E3_inv_B)
                alg3_c_mat = np.dot(Q0, Q1)
                alg3_d_mat = np.dot(Q0, np.dot(P1, Q2))

                decoupled_sys.set_dynamics(ode_a_mat, ode_b_mat, alg1_a_mat, alg1_b_mat, \
                                               alg2_a_mat, alg2_b_mat, alg2_c_mat, alg3_a_mat, \
                                               alg3_b_mat, alg3_c_mat, alg3_d_mat, matrix_c)

                self.decoupled_sys = decoupled_sys
                self.status = 'success'

        return self.decoupled_sys, self.status


class DecouplingAutonomous(object):
    'implement decoupling techniques using admissible projectors for autonomous dae'

    def __init__(self):

        self.decoupled_sys = None    # return decoupled system
        self.status = None    # return status of decoupling process

    def get_decoupled_system(self, auto_dae_automaton):
        'get decoupled system from an dae automaton'

        assert isinstance(auto_dae_automaton, AutonomousDaeAutomation)

        matrix_e = auto_dae_automaton.matrix_e
        matrix_a = auto_dae_automaton.matrix_a
        matrix_c = auto_dae_automaton.matrix_c

        n, _ = matrix_a.shape[0]
        In = np.eye(n, dtype=float)

        adm_projs, e_mu_inv, _ = admissible_projectors(matrix_e, matrix_a)

        if adm_projs == 'error':
            print "\nerror: dae system has index larger than 3"
            self.status = 'error'
        else:
            assert isinstance(adm_projs, list), 'error: not a list of admissible projectors'
            ind = len(adm_projs)

            if ind == 1:

                decoupled_sys = AutonomousDecoupledIndexOne()
                Q0 = adm_projs[0]
                P0 = In - Q0
                A0 = matrix_a
                e_mu_inv_A0 = np.dot(e_mu_inv, A0)

                # ode part
                ode_a_mat = np.dot(P0, e_mu_inv_A0)

                # alg part
                alg_a_mat = np.dot(Q0, e_mu_inv_A0)
                decoupled_sys.set_dynamics(ode_a_mat, alg_a_mat, matrix_c)

                self.decoupled_sys = decoupled_sys
                self.status = 'success'

            elif ind == 2:

                decoupled_sys = AutonomousDecoupledIndexTwo()
                Q0 = adm_projs[0]
                Q1 = adm_projs[1]
                P0 = In - Q0
                P1 = In - Q1
                # ode part
                P0_P1 = np.dot(P0, P1)
                A2 = np.dot(matrix_a, P0_P1)
                E2_inv_A2 = np.dot(e_mu_inv, A2)
                ode_a_mat = np.dot(P0_P1, E2_inv_A2)

                # alg1 part
                alg1_a_mat = np.dot(Q1, E2_inv_A2)

                # alg2 part
                Q0_P1 = np.dot(Q0, P1)
                alg2_a_mat = np.dot(Q0_P1, E2_inv_A2)
                alg2_c_mat = np.dot(Q0, Q1)

                decoupled_sys.set_dynamics(ode_a_mat, alg1_a_mat, alg2_a_mat, alg2_c_mat, matrix_c)

                self.decoupled_sys = decoupled_sys
                self.status = 'success'

            elif ind == 3:

                decoupled_sys = AutonomousDecoupledIndexThree()
                Q0 = adm_projs[0]
                Q1 = adm_projs[1]
                Q2 = adm_projs[2]
                P0 = In - Q0
                P1 = In - Q1
                P2 = In - Q2

                P0_P1_P2 = np.dot(P0, np.dot(P1, P2))
                A3 = np.dot(A0, P0_P1_P2)
                E3_inv_A3 = np.dot(e_mu_inv, A3)
                # ode part
                ode_a_mat = np.dot(P0_P1_P2, E3_inv_A3)

                # alg1 part
                P0_P1_Q2 = np.dot(P0, np.dot(P1, Q2))
                alg1_a_mat = np.dot(P0_P1_Q2, E3_inv_A3)

                # alg2 part
                P0_Q1_P2 = np.dot(P0, np.dot(Q1, P2))
                alg2_a_mat = np.dot(P0_Q1_P2, E3_inv_A3)
                alg2_c_mat = np.dot(P0, np.dot(Q1, Q2))

                # alg3 part
                Q0_P1_P2 = np.dot(Q0, np.dot(P1, P2))
                alg3_a_mat = np.dot(Q0_P1_P2, E3_inv_A3)
                alg3_c_mat = np.dot(Q0, Q1)
                alg3_d_mat = np.dot(Q0, np.dot(P1, Q2))

                decoupled_sys.set_dynamics(ode_a_mat, alg1_a_mat, alg2_a_mat, alg2_c_mat, alg3_a_mat, alg3_c_mat, alg3_d_mat, matrix_c)

                self.decoupled_sys = decoupled_sys
                self.status = 'success'

        return self.decoupled_sys, self.status
