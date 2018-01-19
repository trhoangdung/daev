'''
This module implements decoupling techniques for high-index DAE
Dung Tran: Dec/2017
Update: Jan/2018
'''

import numpy as np
from daev.engine.dae_automaton import DaeAutomation, AutonomousDaeAutomation
from daev.engine.projectors import admissible_projectors, null_space
from daev.engine.set import ReachSet


class DecoupledIndexOne(object):
    'Decoupled system of index-1 dae'

    # index-1 dae can be decoupled into a decoupled system:
    #     dot{x1} = N1 * x1 + M1 * u
    #     x2 = N2 * x1 + M2 * u
    #     y = C(x1 + x2)

    def __init__(self):
        self.name = 'DecoupledIndexOne'
        self.N1 = None    # ode part
        self.M1 = None    # ode part
        self.N2 = None    # algebraic constraint part
        self.M2 = None    # algebraic constraint part
        self.out_matrix_c = None    # output matrix
        self.projectors = None    # projectors in decoupling process, used to check consistent initial condition
        self.x1_init_set_projector = None    # projector to compute the initial set for ode subsystem from the original init set

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

        self.N1 = ode_a_mat
        self.M1 = ode_b_mat
        self.N2 = alg_a_mat
        self.M2 = alg_b_mat
        self.out_matrix_c = c_mat

    def set_projectors(self, projectors_list):
        'store projectors of decoupling process'

        assert isinstance(projectors_list, list)
        assert len(projectors_list) == 1, 'error: invalid projector list'
        self.projectors = projectors_list
        Q0 = projectors_list[0]
        P0 = np.eye(Q0.shape[0]) - Q0
        self.x1_init_set_projector = P0


class AutonomousDecoupledIndexOne(object):
    'AutonomousDecoupled system of index-1 dae'

    # autonomous index-1 dae can be decoupled into a decoupled system:
    #     dot{x1} = N1 * x1
    #     x2 = N2 * x1
    #     y = C(x1 + x2)

    def __init__(self):
        self.name = 'AutonomousDecoupledIndexOne'
        self.N1 = None    # ode part
        self.N2 = None    # algebraic constraint part
        self.reach_set_projector = None    # projector to compute the reachable set the original system
        self.x1_init_set_projector = None    # projector to compute the initial set for ode subsystem from the original init set

        self.out_matrix_c = None    # output matrix
        self.projectors = []    # projectors in decoupling process, used to check consistent initial condition
        self.consistent_matrix = None    # consistent matrix to check consistency of initial condition

    def set_dynamics(self, ode_a_mat, alg_a_mat, c_mat):
        'set dynamics for decoupled system'

        assert isinstance(ode_a_mat, np.ndarray)
        assert isinstance(alg_a_mat, np.ndarray)
        assert isinstance(c_mat, np.ndarray)

        assert ode_a_mat.shape[0] == ode_a_mat.shape[1], 'error: invalid ode_a_mat'
        assert alg_a_mat.shape[0] == alg_a_mat.shape[1], 'error: invalid alg_a_mat'

        assert alg_a_mat.shape[0] == ode_a_mat.shape[0], 'error: inconsistency'
        assert c_mat.shape[1] == ode_a_mat.shape[0], 'error: inconsistency'

        self.N1 = ode_a_mat
        self.N2 = alg_a_mat
        self.reach_set_projector = np.eye(self.N2.shape[0]) + self.N2
        self.out_matrix_c = c_mat

    def set_projectors(self, projectors_list):
        'store projectors of decoupling process'

        assert isinstance(projectors_list, list)
        assert len(projectors_list) == 1, 'error: invalid projector list'
        self.projectors = projectors_list
        Q0 = projectors_list[0]
        P0 = np.eye(Q0.shape[0]) - Q0
        self.x1_init_set_projector = P0

    def get_consistent_matrix(self):
        'construct consistent matrix to check consistency of initial condition'

        assert self.projectors != [], 'error: empty set of projectors'
        assert self.ode_matrix_a is not None, 'error: empty decoupled system'

        Q0 = self.projectors[0]
        P0 = np.eye(Q0.shape[0]) - Q0
        # Q0 - N2 * P0
        self.consistent_matrix = Q0 - np.dot(self.N2, P0)

        return self.consistent_matrix

    def check_consistency(self, init_set):
        'check consistency of initial condition'

        assert isinstance(init_set, ReachSet)

        if self.consistent_matrix is None:
            self.get_consistent_matrix()
        else:
            if np.linalg.norm(np.dot(self.consistent_matrix, init_set.S)) > 1e-6:
                print "\nError: initial condition is not consistent"
                consistency = True
            else:
                consistency = False

        return consistency

    def generate_consistent_init_set_basic_matrix(self):
        'automatically generate a consistent basic matrix for the initial set of state'

        if self.consistent_matrix is None:
            self.get_consistent_matrix()
        else:
            basic_matrix, _ = null_space(self.consistent_matrix)

        return basic_matrix


class DecoupledIndexTwo(object):
    'Decoupled system of index-2 dae'

    # index-2 dae can be decoupled into a system as follows:
    #    dot{x1} = N1 * x1 + M1 * u    (ode part)
    #    x2 = N2 * x1 + M2 * u         (algebraic constraint part 1)
    #    x3 = N3 * x1 + M3 * u + L3 * dot{x2}    (algebraic constraint part 2)
    #    y = c(x1 + x2 + x3)

    def __init__(self):
        self.name = 'DecoupledIndexTwo'
        self.N1 = None    # ode part
        self.M1 = None    # ode part
        self.N2 = None    # algebraic constraints part 1
        self.M2 = None    # algebraic constraints part 1
        self.N3 = None    # algebraic constraints part 2
        self.M3 = None    # algebraic constraints part 2
        self.L3 = None    # algebraic constraints part 2
        self.out_matrix_c = None     # output matrix
        self.projectors = []    # projectors in decoupling process, used to check consistent initial condition
        self.x1_init_set_projector = None    # projector to compute the initial set for ode subsystem from the original init set

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

        self.N1 = ode_a_mat
        self.M1 = ode_b_mat
        self.N2 = alg1_a_mat
        self.M2 = alg1_b_mat
        self.N3 = alg2_a_mat
        self.M3 = alg2_b_mat
        self.L3 = alg2_c_mat
        self.out_matrix_c = c_mat

    def set_projectors(self, projectors_list):
        'store projectors of decoupling process'

        assert isinstance(projectors_list, list)
        assert len(projectors_list) == 2, 'error: invalid projector list'
        self.projectors = projectors_list
        Q0 = projectors_list[0]
        Q1 = projectors_list[1]
        P0 = np.eye(Q0.shape[0]) - Q0
        P1 = np.eye(Q1.shape[0]) - Q1
        self.x1_init_set_projector = np.dot(P0, P1)


class AutonomousDecoupledIndexTwo(object):
    'AutonomousDecoupled system of index-2 dae'

    # autonomous index-2 dae can be decoupled into a system as follows:
    #    dot{x1} = N1 * x1     (ode part)
    #    x2 = N2 * x1          (algebraic constraint part 1)
    #    x3 = N3 * x1 + L3 * dot{x2}    (algebraic constraint part 2)
    #    y = C(x1 + x2 + x3)

    def __init__(self):
        self.name = 'AutonomousDecoupledIndexTwo'
        self.N1 = None    # ode part
        self.N2 = None    # algebraic constraints part 1
        self.N3 = None    # algebraic constraints part 2
        self.L3 = None    # algebraic constraints part 2
        self.reach_set_projector = None    # projector to compute the reachable set the original system
        self.x1_init_set_projector = None    # projector to compute the initial set for ode subsystem from the original init set

        self.out_matrix_c = None     # output matrix
        self.projectors = []    # projectors in decoupling process, used to check consistent initial condition
        self.consistent_matrix = None    # consistent matrix to check consistency of initial condition

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

        self.N1 = ode_a_mat
        self.N2 = alg1_a_mat
        self.N3 = alg2_a_mat
        self.L3 = alg2_c_mat
        self.reach_set_projector = np.eye(self.N1.shape[0]) + self.N2 + self.N3 + np.dot(self.L3, np.dot(self.N2, self.N1))

        self.out_matrix_c = c_mat

    def set_projectors(self, projectors_list):
        'store projectors of decoupling process'

        assert isinstance(projectors_list, list)
        assert len(projectors_list) == 2, 'error: invalid projector list'
        self.projectors = projectors_list
        Q0 = projectors_list[0]
        Q1 = projector_list[1]
        P0 = np.eye(Q0.shape[0]) - Q0
        P1 = np.eye(Q1.shape[0]) - Q1
        self.x1_init_set_projector = np.dot(P0, P1)

    def get_consistent_matrix(self):
        'construct consistent matrix to check the consistency of the initial condition'

        assert self.projectors is not None, 'error: empty projectors list'
        assert self.ode_matrix_a is not None, 'error: empty decoupled system'

        Q0 = self.projectors[0]
        P0 = np.eye(Q0.shape[0]) - Q0
        Q1 = self.projectors[1]
        P1 = np.eye(Q1.shape[0]) - Q1

        # P0 * Q1 - N2 * P0 * P1
        C1 = np.dot(P0, Q1) - np.dot(self.N2, np.dot(P0, P1))
        # Q0 - (N3 + L3 * N2 * N1) * P0 * P1
        C2 = Q0 - np.dot(self.N3 + np.dot(self.L3, np.dot(self.N2, self.N1)), np.dot(P0, P1))
        self.consistent_matrix = np.vstack((C1, C2))

        return self.consistent_matrix

    def check_consistency(self, init_set):
        'check consistency of initial condition'

        assert isinstance(init_set, ReachSet)

        if self.consistent_matrix is None:
            self.get_consistent_matrix()
        else:
            if np.linalg.norm(np.dot(self.consistent_matrix, init_set.S)) > 1e-6:
                print "\nError: initial condition is not consistent"
                consistency = True
            else:
                consistency = False

        return consistency

    def generate_consistent_init_set_basic_matrix(self):
        'automatically generate a consistent basic matrix for the initial set of state'

        if self.consistent_matrix is None:
            self.get_consistent_matrix()
        else:
            basic_matrix, _ = null_space(self.consistent_matrix)

        return basic_matrix


class DecoupledIndexThree(object):
    'Decoupled system of index-3 dae'

    # index-3 dae can be decoupled into a system as follows:
    #    ode:    dot{x1} = N1 * x1 + M1 * u
    #    alg1:    x2 = N2 * x1 + M2 * u
    #    alg2:    x3 = N3 * x1 + M3 * u + L3 * dot{x2}
    #    alg3:    x4 = N4 * x1 + M4 * u + L4 * dot{x3} + Z4 * dot{x2}
    #    output:  y = C(x1 + x2 + x3 + x4)

    def __init__(self):
        self.name = 'DecoupledIndexThree'
        self.N1 = None    # ode part
        self.M1 = None    # ode part
        self.N2 = None    # alg1 part
        self.M2 = None    # alg1 part
        self.N3 = None    # alg2 part
        self.M3 = None    # alg2 part
        self.L3 = None    # alg2 part
        self.N4 = None    # alg3 part
        self.M4 = None    # alg3 part
        self.L4 = None    # alg3 part
        self.Z4 = None    # alg3 part
        self.out_matrix_c = None    # output matrix
        self.projectors = []    # projectors in decoupling process, used to check consistent initial condition
        self.x1_init_set_projector = None    # projector to compute the initial set for ode subsystem from the original init set

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

        self.N1 = ode_a_mat
        self.M1 = ode_b_mat
        self.N2 = alg1_a_mat
        self.M2 = alg1_b_mat
        self.N3 = alg2_a_mat
        self.M3 = alg2_b_mat
        self.L3 = alg2_c_mat
        self.N4 = alg3_a_mat
        self.M4 = alg3_b_mat
        self.L4 = alg3_c_mat
        self.Z4 = alg3_d_mat
        self.out_matrix_c = c_mat

    def set_projectors(self, projectors_list):
        'store projectors of decoupling process'

        assert isinstance(projectors_list, list)
        assert len(projectors_list) == 3, 'error: invalid projector list'
        self.projectors = projectors_list
        Q0 = projectors_list[0]
        Q1 = projectors_list[1]
        Q2 = projectors_list[2]
        P0 = np.eye(Q0.shape[0]) - Q0
        P1 = np.eye(Q1.shape[0]) - Q1
        P2 = np.eye(Q2.shape[0]) - Q2

        self.x1_init_set_projector = np.dot(P0, np.dot(P1, P2))


class AutonomousDecoupledIndexThree(object):
    'Autonomous Decoupled system of index-3 dae'

    # autonomous index-3 dae can be decoupled into a system as follows:
    #    ode:    dot{x1} = N1 * x1
    #    alg1:    x2 = N2 * x1
    #    alg2:    x3 = N3 * x1 + L3 * dot{x2}
    #    alg3:    x4 = N4 * x1 + L4 * dot{x3} + Z4 * dot{x2}
    #    output:  y = C(x1 + x2 + x3 + x4)

    def __init__(self):
        self.name = 'AutonomousDecoupledIndexThree'
        self.N1 = None    # ode part
        self.N2 = None    # alg1 part
        self.N3 = None    # alg2 part
        self.L3 = None    # alg2 part
        self.N4 = None    # alg3 part
        self.L4 = None    # alg3 part
        self.Z4 = None    # alg3 part
        self.reach_set_projector = None    # projector to compute the reachable set the original system
        self.x1_init_set_projector = None    # projector to compute the initial set for ode subsystem from the original init set

        self.out_matrix_c = None    # output matrix
        self.projectors = []    # projectors in decoupling process, used to check consistent initial condition
        self.consistent_matrix = None    # consistent matrix to check consistency of initial condition

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

        self.N1 = ode_a_mat

        self.N2 = alg1_a_mat
        self.N3 = alg2_a_mat
        self.L3 = alg2_c_mat
        self.N4 = alg3_a_mat
        self.L4 = alg3_c_mat
        self.Z4 = alg3_d_mat

        # compute reach set projector
        In = np.eye(self.N2.shape[0])
        L3N2N1 = np.dot(self.L3, np.dot(self.N2, self.N1))
        L4N3N1 = np.dot(self.L4, np.dot(self.N3, self.N1))
        L4L3N2N1N1 = np.dot(self.L4, np.dot(self.L3, np.dot(self.N2, np.dot(self.N1, self.N1))))
        Z4N2N1 = np.dot(self.Z4, np.dot(self.N2, self.N1))
        self.reach_set_projector = In + self.N2 + self.N3 + self.N4 + L3N2N1 + L4N3N1 + L4L3N2N1N1 + Z4N2N1
        self.out_matrix_c = c_mat

    def set_projectors(self, projectors_list):
        'store projectors of decoupling process'

        assert isinstance(projectors_list, list)
        assert len(projectors_list) == 3, 'error: invalid projector list'
        self.projectors = projectors_list
        Q0 = projectors_list[0]
        Q1 = projectors_list[1]
        Q2 = projectors_list[2]
        P0 = np.eye(Q0.shape[0]) - Q0
        P1 = np.eye(Q1.shape[0]) - Q1
        P2 = np.eye(Q2.shape[0]) - Q2

        self.x1_init_set_projector = np.dot(P0, np.dot(P1, P2))

    def get_consistent_matrix(self):
        'construct consistent matrix to check the consistency of the initial condition'

        assert self.projectors is not None, 'error: empty projectors list'
        assert self.ode_matrix_a is not None, 'error: empty decoupled system'

        Q0 = self.projectors[0]
        P0 = np.eye(Q0.shape[0]) - Q0
        Q1 = self.projectors[1]
        P1 = np.eye(Q1.shape[0]) - Q1
        Q2 = self.projectors[2]
        P2 = np.eye(Q2.shape[0]) - Q2


        # P0 * P1 * Q2 - N2 * P0 * P1 * P2
        C1 = np.dot(P0, np.dot(P1, Q2)) - np.dot(self.N2, np.dot(P0, np.dot(P1, P2)))
        # P0 * Q1 - (N3 + L3 * N2 * N1) * P0 * P1 * P2
        C2 = np.dot(P0, Q1) - np.dot(self.N3 + np.dot(self.L3, np.dot(self.N2, self.N1)), np.dot(P0, np.dot(P1, P2)))

        # Q0 - [N4 + L4 (N3 * N1 + L3 * N2 * N1^2) + Z4*N2*N1)]P0*P1*P2

        P0P1P2 = np.dot(P0, np.dot(P1, P2))
        Z4N2N1 = np.dot(self.Z4, np.dot(self.N2, self.N1))
        L3N2N1N1 = np.dot(self.L3, np.dot(self.N2, np.dot(self.N1, self.N1)))
        N3N1 = np.dot(self.N3, self.N1)
        C3 = Q0 - np.dot(self.N4 + np.dot(self.L4, N3N1 + L3N2N1N1) + Z4N2N1, P0P1P2)
        self.consistent_matrix = np.vstack((C1, C2, C3))

        return self.consistent_matrix

    def check_consistency(self, init_set):
        'check consistency of initial condition'

        assert isinstance(init_set, ReachSet)

        if self.consistent_matrix is None:
            self.get_consistent_matrix()
        else:
            if np.linalg.norm(np.dot(self.consistent_matrix, init_set.S)) > 1e-6:
                print "\nError: initial condition is not consistent"
                consistency = True
            else:
                consistency = False

        return consistency

    def generate_consistent_init_set_basic_matrix(self):
        'automatically generate a consistent basic matrix for the initial set of state'

        if self.consistent_matrix is None:
            self.get_consistent_matrix()
        else:
            basic_matrix, _ = null_space(self.consistent_matrix)

        return basic_matrix


class Decoupling(object):
    'implement decoupling techniques using admissible projectors'

    def __init__(self):

        self.decoupled_sys = None    # return decoupled system
        self.status = None    # return status of decoupling process

    def get_decoupled_system(self, dae_automaton):
        'get decoupled system from an dae automaton'

        assert isinstance(dae_automaton, DaeAutomation)

        matrix_e = dae_automaton.matrix_e.todense()
        matrix_a = dae_automaton.matrix_a.todense()
        matrix_b = dae_automaton.matrix_b.todense()
        print "\nmatrix_b = {}".format(matrix_b)
        matrix_c = dae_automaton.matrix_c.todense()
        print "\nmatrix_c = {}".format(matrix_c)

        n = matrix_a.shape[0]
        In = np.eye(n, dtype=float)

        adm_projs, E_list, A_list, _ = admissible_projectors(matrix_e, matrix_a)

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
                E1_inv = np.linalg.inv(E_list[1])
                A0 = A_list[0]
                E1_inv_A0 = np.dot(E1_inv, A0)
                E1_inv_B = np.dot(E1_inv, matrix_b)
                # ode part
                N1 = np.dot(P0, E1_inv_A0)
                M1 = np.dot(P0, E1_inv_B)

                # alg part
                N2 = np.dot(Q0, E1_inv_A0)
                M2 = np.dot(Q0, E1_inv_B)

                decoupled_sys.set_dynamics(N1, M1, N2, M2, matrix_c)
                decoupled_sys.set_projectors(adm_projs)

                self.decoupled_sys = decoupled_sys
                self.status = 'success'

            elif ind == 2:

                decoupled_sys = DecoupledIndexTwo()
                Q0 = adm_projs[0]
                Q1 = adm_projs[1]
                P0 = In - Q0
                P1 = In - Q1

                P0_P1 = np.dot(P0, P1)
                E2_inv = np.linalg.inv(E_list[2])
                A2 = A_list[2]
                E2_inv_A2 = np.dot(E2_inv, A2)
                E2_inv_B = np.dot(E2_inv, matrix_b)

                # ode part
                N1 = np.dot(P0_P1, E2_inv_A2)
                M1 = np.dot(P0_P1, E2_inv_B)

                # alg1 part
                N2 = np.dot(P0, np.dot(Q1, E2_inv_A2))
                M2 = np.dot(P0, np.dot(Q1, E2_inv_B))

                # alg2 part
                Q0_P1 = np.dot(Q0, P1)
                N3 = np.dot(Q0_P1, E2_inv_A2)
                M3 = np.dot(Q0_P1, E2_inv_B)
                L3 = np.dot(Q0, Q1)

                decoupled_sys.set_dynamics(N1, M1, N2, M2, N3, M3, L3, matrix_c)

                decoupled_sys.set_projectors(adm_projs)
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
                A3 = A_list[3]
                E3_inv = np.linalg.inv(E_list[3])
                E3_inv_A3 = np.dot(E3_inv, A3)
                E3_inv_B = np.dot(E3_inv, matrix_b)
                # ode part
                N1 = np.dot(P0_P1_P2, E3_inv_A3)
                M1 = np.dot(P0_P1_P2, E3_inv_B)

                # alg1 part
                P0_P1_Q2 = np.dot(P0, np.dot(P1, Q2))
                N2 = np.dot(P0_P1_Q2, E3_inv_A3)
                M2 = np.dot(P0_P1_Q2, E3_inv_B)

                # alg2 part
                P0_Q1_P2 = np.dot(P0, np.dot(Q1, P2))
                N3 = np.dot(P0_Q1_P2, E3_inv_A3)
                M3 = np.dot(P0_Q1_P2, E3_inv_B)
                L3 = np.dot(P0, np.dot(Q1, Q2))

                # alg3 part
                Q0_P1_P2 = np.dot(Q0, np.dot(P1, P2))
                N4 = np.dot(Q0_P1_P2, E3_inv_A3)
                M4 = np.dot(Q0_P1_P2, E3_inv_B)
                L4 = np.dot(Q0, Q1)
                Z4 = np.dot(Q0, np.dot(P1, Q2))

                decoupled_sys.set_dynamics(N1, M1, N2, M2, N3, M3, L3, N4, M4, L4, Z4, matrix_c)
                decoupled_sys.set_projectors(adm_projs)
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

        matrix_e = auto_dae_automaton.matrix_e.todense()
        matrix_a = auto_dae_automaton.matrix_a.todense()
        matrix_c = auto_dae_automaton.matrix_c.todense()

        n = matrix_a.shape[0]
        In = np.eye(n, dtype=float)

        adm_projs, E_list, A_list, _ = admissible_projectors(matrix_e, matrix_a)

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
                A0 = A_list[0]
                E1_inv = np.linalg.inv(E_list[1])
                E1_inv_A0 = np.dot(E1_inv, A0)

                N1 = np.dot(P0, E1_inv_A0)    # ode part
                N2 = np.dot(Q0, E1_inv_A0)    # alg part

                decoupled_sys.set_dynamics(N1, N2, matrix_c)
                decoupled_sys.set_projectors(adm_projs)
                self.decoupled_sys = decoupled_sys
                self.status = 'success'

            elif ind == 2:

                decoupled_sys = AutonomousDecoupledIndexTwo()
                Q0 = adm_projs[0]
                Q1 = adm_projs[1]
                P0 = In - Q0
                P1 = In - Q1

                P0_P1 = np.dot(P0, P1)
                A2 = A_list[2]
                E2_inv = np.linalg.inv(E_list[2])
                E2_inv_A2 = np.dot(E2_inv, A2)

                N1 = np.dot(P0_P1, E2_inv_A2)    # ode part
                N2 = np.dot(P0, np.dot(Q1, E2_inv_A2))    # alg1 part

                # alg2 part
                Q0_P1 = np.dot(Q0, P1)
                N3 = np.dot(Q0_P1, E2_inv_A2)
                L3 = np.dot(Q0, Q1)

                decoupled_sys.set_dynamics(N1, N2, N3, L3, matrix_c)
                decoupled_sys.set_projectors(adm_projs)
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
                A0 = matrix_a

                P0_P1_P2 = np.dot(P0, np.dot(P1, P2))
                A3 = A_list[3]
                E3_inv = np.linalg.inv(E_list[3])
                E3_inv_A3 = np.dot(E3_inv, A3)

                # ode part
                N1 = np.dot(P0_P1_P2, E3_inv_A3)

                # alg1 part
                P0_P1_Q2 = np.dot(P0, np.dot(P1, Q2))
                N2 = np.dot(P0_P1_Q2, E3_inv_A3)

                # alg2 part
                P0_Q1_P2 = np.dot(P0, np.dot(Q1, P2))
                N3 = np.dot(P0_Q1_P2, E3_inv_A3)
                L3 = np.dot(P0, np.dot(Q1, Q2))

                # alg3 part
                Q0_P1_P2 = np.dot(Q0, np.dot(P1, P2))
                N4 = np.dot(Q0_P1_P2, E3_inv_A3)
                L4 = np.dot(Q0, Q1)
                Z4 = np.dot(Q0, np.dot(P1, Q2))

                decoupled_sys.set_dynamics(N1, N2, N3, L3, N4, L4, Z4, matrix_c)
                decoupled_sys.set_projectors(adm_projs)
                self.decoupled_sys = decoupled_sys
                self.status = 'success'

        return self.decoupled_sys, self.status
