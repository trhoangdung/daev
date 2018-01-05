'''
This modules implements simulation-based reachability analysis for decoupled dae systems
Dung Tran: Dec/2017
'''

import time
import numpy as np
from daev.engine.set import ReachSet
from daev.engine.decoupling import AutonomousDecoupledIndexOne, AutonomousDecoupledIndexTwo, AutonomousDecoupledIndexThree
from daev.engine.dae_automaton import DaeAutomation
from daev.engine.projectors import null_space
from scipy.integrate import ode


class ReachSetAssembler(object):
    'implements rechable set computation for odes, daes'

    @staticmethod
    def ode_sim(matrix_a, init_vec, totime, num_steps, solver_name):
        'compute simulation trace of an ode'
        # solvers can be selected from the list = ['vode', 'zvode',  'Isoda',
        # 'dopri5', 'dop853']

        start = time.time()
        assert isinstance(
            matrix_a, np.ndarray), 'error: matrix_a is not ndarray'
        if solver_name != 'vode' and solver_name != 'zvode' and solver_name != 'Isoda' and solver_name != 'dopri5' and solver_name != 'dop853':
            raise ValueError('error: invalid solver name')

        n = matrix_a.shape[0]

        def fun(t, y):
            'derivative function'
            rv = np.dot(matrix_a, y)
            return rv

        t0 = 0.0
        t = np.linspace(t0, totime, num_steps + 1)

        solver = ode(fun)
        solver.set_integrator(solver_name)
        solver.set_initial_value(init_vec, t0)
        sol = np.empty((num_steps + 1, n))
        sol[0] = init_vec

        k = 1
        while solver.successful() and solver.t < totime:
            solver.integrate(t[k])
            sol[k] = solver.y
            k += 1

        runtime = time.time() - start
        return sol, runtime

    @staticmethod
    def reach_autonomous_ode(matrix_a, init_reachset,
                             totime, num_steps, solver_name):
        'compute reachable set of automnous linear ode: \dot{x} = Ax'

        # compute reachable set using simulation
        # solvers can be selected from the list = ['vode', 'zvode',  'Isoda',
        # 'dopri5', 'dop853']
        start = time.time()
        assert isinstance(
            matrix_a, np.ndarray) and matrix_a.shape[0] == matrix_a.shape[1], 'error: invalid matrix a'
        assert isinstance(init_reachset, ReachSet)
        assert matrix_a.shape[0] == init_reachset.S.shape[0], 'error: inconsistent matrix_a and initial reach set'
        assert isinstance(totime, float) and totime > 0, 'error: invalid time'
        assert isinstance(
            num_steps, int) and num_steps >= 0, 'error: invalid number of steps'

        matrix_S = init_reachset.S
        alpha_min = init_reachset.alpha_min_vec
        alpha_max = init_reachset.alpha_max_vec
        n, k = matrix_S.shape
        sol_list = []

        for j in xrange(0, k):
            init_vec = matrix_S[:, j]
            sol, _ = ReachSetAssembler().ode_sim(
                matrix_a, init_vec, totime, num_steps, solver_name)
            sol_list.append(sol)

        reach_set_list = []
        for i in xrange(0, num_steps):
            reach_set = ReachSet()
            s_mat = np.empty((n, k))
            for j in xrange(0, k):
                sol = sol_list[j]
                s_mat[:, j] = np.transpose(sol[i])

            reach_set.set_params(s_mat, alpha_min, alpha_max)
            reach_set_list.append(reach_set)

        runtime = time.time() - start

        return reach_set_list, runtime

    @staticmethod
    def check_consistency(decoupled_sys, init_reachset):
        'check consistency of initial condition'

        assert isinstance(decoupled_sys, AutonomousDecoupledIndexOne) or \
            isinstance(decoupled_sys, AutonomousDecoupledIndexTwo) or \
            isinstance(
            decoupled_sys, AutonomousDecoupledIndexThree), 'error: invalid decoupled system'

        assert isinstance(init_reachset, ReachSet)
        S0 = init_reachset.S
        consistency = None    # True/False

        if decoupled_sys.name == 'AutonomousDecoupledIndexOne':

            # initset is X(0) = S(0) * alpha
            # for index-1 decoupled autonomous dae
            # consistent condition is : x2(0) = A2 * x1(0), x2 = Q * x, x1 = P * x
            # thus, we need: Q * S(0) = A2 * P * S(0) or S(0) is null-space of
            # (Q - A2 * P)
            Q0 = decoupled_sys.projectors[0]
            P0 = np.eye(Q0.shape[0], dtype=float) - Q0
            A2 = decoupled_sys.alg_matrix_a
            V = Q0 - np.dot(A2, P0)
            if np.linalg.norm(V, S0) > 1e-6:
                print "\ninconsistent initial condition"
                consistency = False
            else:
                print "\nconsistent initial condition"
                consistency = True

        elif decoupled_sys.name == 'AutonomousDecoupledIndexTwo':

            # for index-2 decoupled autonomous dae
            # the consistent condition is: x2(0) = A2 * x1(0), x3(0) = A3 * x1(0) + C3 * dot{x2}(0)
            # where:     x1 = P0 * P1 * x, x2 = P0 * Q1 * x, x3 = Q0 * x
            # thus the consistent condition is:
            #            1) P0 * Q1 * S(0) = A2 * P0 * P1 * S(0)
            #            2) Q0 * S(0) = (A3 + C3 * A2 * A1) * P0 * P1 * S(0)
            # or S(0) is a null space of V, where V = [P0 * Q1 - A2 * P0 * P1;
            # Q0 - (A3 + C3 * A2 * A1) * P0 * P1]

            Q0 = decoupled_sys.projectors[0]
            Q1 = decoupled_sys.projectors[1]
            P0 = np.eye(Q0.shape[0], dtype=float) - Q0
            P1 = np.eye(Q1.shape[0], dtype=float) - Q1
            A1 = decoupled_sys.ode_matrix_a
            A2 = decoupled_sys.alg1_matrix_a
            A3 = decoupled_sys.alg2_matrix_a
            C3 = decoupled_sys.alg2_matrix_c
            V1 = np.dot(P0, Q1) - np.dot(A2, np.dot(P0, P1))
            V2 = Q0 - np.dot(A3, np.dot(P0, P1)) - np.dot(C3,
                                                          np.dot(A2, np.dot(A1, np.dot(P0, P1))))
            V = np.vstack((V1, V2))
            if np.linalg.norm(V, S0) > 1e-6:
                print "\ninconsistent initial condition"
                consistency = False
            else:
                print "\nconsistent initial condition"
                consistency = True

        elif decoupled_sys.name == 'AutonomousDecoupledIndexThree':

            # for index-3 decoupled autonomous dae
            # the consistent condition is:
            #            1) x2(0) = A2 * x1(0)
            #            2) x3(0) = A3 * x1(0) + C3 * dot{x2}(0)
            #            3) x4(0) = A4 * x1(0) + C4 * dot{x3}(0) + D4 * dot{x2}(0)
            # where: x1 = P0 * P1 * P2 * x, x2 = P0 * P1 * Q2 * x, x3 = P0 * Q1, x4 = Q0
            # thus, the consistent condition becomes:
            #            1) P0 * P1 * Q2 * S(0) = A2 * P0 * P1 * P2 * S(0)
            #            2) P0 * Q1 * S(0) = (A3 + C3 * A2 * A1) * P0 * P1 * P2 * S(0)
            # 3) Q0 * S(0) = (A4 + C4 * A3 * A1 + C3 * A2 * A1 * A1 + D4 * A2 *
            # A1) * P0 * P1 * P2 * S(0)

            Q0 = decoupled_sys.projectors[0]
            Q1 = decoupled_sys.projectors[1]
            Q2 = decoupled_sys.projectors[2]
            n = Q0.shape[0]
            In = np.eye(n, dtype=float)
            P0 = In - Q0
            P1 = In - Q1
            P2 = In - Q2
            A1 = decoupled_sys.ode_matrix_a
            A2 = decoupled_sys.alg1_matrix_a
            A3 = decoupled_sys.alg2_matrix_a
            C3 = decoupled_sys.alg2_matrix_c
            A4 = decoupled_sys.alg3_matrix_a
            C4 = decoupled_sys.alg3_matrix_c
            D4 = decoupled_sys.alg3_matrix_d
            P0_P1_P2 = np.dot(P0, P1, P2)
            V1 = np.dot(P0, np.dot(P1, Q2)) - np.dot(A2, P0_P1_P2)
            V2 = np.dot(P0, Q1) - np.dot(A3 +
                                         np.dot(C3, np.dot(A2, A1)), P0_P1_P2)
            V30 = A4 + np.dot(C4, np.dot(A3, A1) + \
                                  np.dot(C3, np.dot(A2, np.dot(A1, A1)))) + \
                                  np.dot(D4, np.dot(A2, A1))
            V3 = Q0 - np.dot(V30, P0_P1_P2)

            V = np.vstack((V1, V2, V3))
            if np.linalg.norm(V, S0) > 1e-6:
                print "\ninconsistent initial condition"
                consistency = False
            else:
                print "\nconsistent initial condition"
                consistency = True
        else:
            print "\nerror: unknown decoupled system"

        return consistency

    @staticmethod
    def reach_autonomous_dae(decoupled_sys, init_reachset, totime, num_steps, solver_name):
        'compute reachable set for decoupled dae'

        start = time.time()
        assert isinstance(decoupled_sys, AutonomousDecoupledIndexOne) or \
            isinstance(decoupled_sys, AutonomousDecoupledIndexTwo) or \
            isinstance(decoupled_sys, AutonomousDecoupledIndexThree), 'error: invalid decoupled system'

        assert isinstance(init_reachset, ReachSet)

        x_reach_set_list = []
        consistency = ReachSetAssembler().check_consistency(decoupled_sys, init_reachset)

        if consistency:

            if decoupled_sys.name == 'AutonomousDecoupledIndexOne':
                A1 = decoupled_sys.ode_matrix_a
                A2 = decoupled_sys.alg_matrix_a
                Q0 = decoupled_sys.projectors[0]
                P0 = np.eye(Q0.shape[0], dtype=float) - Q0
                x1_init_reachset = init_reachset.multiply(P0)
                x1_reach_set_list, _ = ReachSetAssembler().reach_autonomous_ode(
                    A1, x1_init_reachset, totime, num_steps, solver_name)
                x2_reach_set_list = []
                n = len(x1_init_reachset)
                for i in xrange(0, n):
                    x2_reach_set_list.append(x1_reach_set_list[i].multiply(A2))
                    # x = x1 + x2
                    x_reach_set_list.append(x1_reach_set_list[i].add(x2_reach_set_list[i]))

            elif decoupled_sys.name == 'AutonomousDecoupledIndexTwo':
                A1 = decoupled_sys.ode_matrix_a
                A2 = decoupled_sys.alg1_matrix_a
                A3 = decoupled_sys.alg2_matrix_a
                C3 = decoupled_sys.alg2_matrix_c
                Q0 = decoupled_sys.projectors[0]
                Q1 = decoupled_sys.projectors[1]
                P0 = np.eye(Q0.shape[0], dtype=float) - Q0
                P1 = np.eye(Q1.shape[0], dtype=float) - Q1

                x1_init_reachset = init_reachset.multiply(np.dot(P0, P1))
                x1_reach_set_list, _ = ReachSetAssembler().reach_autonomous_ode(
                    A1, x1_init_reachset, totime, num_steps, solver_name)
                x2_reach_set_list = []
                x3_reach_set_list = []
                n = len(x1_init_reachset)
                for i in xrange(0, n):
                    x2_reach_set_list.append(x1_reach_set_list[i].multiply(A2))
                    # x3 = A3*x1 + C3 * dot{x2} = A3 * x1 + C3 * A2 \dot{x1}
                    #    = (A3 + C3 * A2 * A1) * x1
                    x3_reach_set_list.append(x1_reach_set_list[i].multiply(A3 + np.dot(C3, np.dot(A2, A1))))
                    # x = x1 + x2 + x3
                    x_reach_set_list.append(x3_reach_set_list[i].add(x2_reach_set_list[i].add(x1_reach_set_list[i])))

            elif decoupled_sys.name == 'AutonomousDecoupledIndexThree':

                A1 = decoupled_sys.ode_matrix_a
                A2 = decoupled_sys.alg1_matrix_a
                A3 = decoupled_sys.alg2_matrix_a
                C3 = decoupled_sys.alg2_matrix_c
                A4 = decoupled_sys.alg3_matrix_a
                C4 = decoupled_sys.alg3_matrix_c
                D4 = decoupled_sys.alg3_matrix_d
                Q0 = decoupled_sys.projectors[0]
                Q1 = decoupled_sys.projectors[1]
                Q2 = decoupled_sys.projectors[2]
                P0 = np.eye(Q0.shape[0], dtype=float) - Q0
                P1 = np.eye(Q1.shape[0], dtype=float) - Q1
                P2 = np.eye(Q2.shape[0], dtype=float) - Q2

                x1_init_reachset = init_reachset.multiply(np.dot(P0, np.dot(P1, P2)))
                x1_reach_set_list, _ = ReachSetAssembler().reach_autonomous_ode(
                    A1, x1_init_reachset, totime, num_steps, solver_name)
                x2_reach_set_list = []
                x3_reach_set_list = []
                x4_reach_set_list = []
                n = len(x1_init_reachset)

                V3 = A3 + np.dot(C3, np.dot(A2, A1))
                V4 = A4 + np.dot(C4, np.dot(A3, A1) + \
                      np.dot(C3, np.dot(A2, np.dot(A1, A1)))) + \
                      np.dot(D4, np.dot(A2, A1))
                for i in xrange(0, n):
                    x2_reach_set_list.append(x1_reach_set_list[i].multiply(A2))
                    # x3 = A3*x1 + C3 * dot{x2} = A3 * x1 + C3 * A2 \dot{x1}
                    #    = (A3 + C3 * A2 * A1) * x1
                    x3_reach_set_list.append(x1_reach_set_list[i].multiply(V3))
                    # x4 = A4 * x1 + C4 * dot{x3} + D4 * dot{x2}
                    x4_reach_set_list.append(x1_reach_set_list[i].multiply(V4))
                    # x = x1 + x2 + x3 + x4
                    x_reach_set_list.append(x4_reach_set_list[i].add(x3_reach_set_list[i].add(x2_reach_set_list[i].add(x1_reach_set_list[i]))))

        else:
            print "\nerror: inconsistent initial condition"

        runtime = time.time() - start

        return x_reach_set_list, runtime

    @staticmethod
    def generate_consistent_init_condition(decoupled_sys):
        'generate a space for consistent initial condition'

        start = time.time()
        assert isinstance(decoupled_sys, AutonomousDecoupledIndexOne) or \
            isinstance(decoupled_sys, AutonomousDecoupledIndexTwo) or \
            isinstance(decoupled_sys, AutonomousDecoupledIndexThree)

        S0 = None
        if decoupled_sys.name == 'AutonomousDecoupledIndexOne':

            # initset is X(0) = S(0) * alpha
            # for index-1 decoupled autonomous dae
            # consistent condition is : x2(0) = A2 * x1(0), x2 = Q * x, x1 = P * x
            # thus, we need: Q * S(0) = A2 * P * S(0) or S(0) is null-space of
            # (Q - A2 * P)

            Q0 = decoupled_sys.projectors[0]
            n = Q0.shape[0]
            In = np.eye(n, dtype=float)
            P0 = In - Q0
            A2 = decoupled_sys.alg_matrix_a
            V = Q0 - np.dot(A2, P0)
            # consistent initial condition for index-1 autonomous dae is S(0) =
            # null_V
            S0, _ = null_space(V)

        elif decoupled_sys.name == 'AutonomousDecoupledIndexTwo':

            # for index-2 decoupled autonomous dae
            # the consistent condition is: x2(0) = A2 * x1(0), x3(0) = A3 * x1(0) + C3 * dot{x2}(0)
            # where:     x1 = P0 * P1 * x, x2 = P0 * Q1 * x, x3 = Q0 * x
            # thus the consistent condition is:
            #            1) P0 * Q1 * S(0) = A2 * P0 * P1 * S(0)
            #            2) Q0 * S(0) = (A3 + C3 * A2 * A1) * P0 * P1 * S(0)
            # or S(0) is a null space of V, where V = [P0 * Q1 - A2 * P0 * P1;
            # Q0 - (A3 + C3 * A2 * A1) * P0 * P1]

            Q0 = decoupled_sys.projectors[0]
            Q1 = decoupled_sys.projectors[1]
            n = Q0.shape[0]
            In = np.eye(n, dtype=float)
            P0 = In - Q0
            P1 = In - Q1
            A1 = decoupled_sys.ode_matrix_a
            A2 = decoupled_sys.alg1_matrix_a
            A3 = decoupled_sys.alg2_matrix_a
            C3 = decoupled_sys.alg2_matrix_c
            V1 = np.dot(P0, Q1) - np.dot(A2, np.dot(P0, P1))
            V2 = Q0 - np.dot(A3, np.dot(P0, P1)) - np.dot(C3,
                                                          np.dot(A2, np.dot(A1, np.dot(P0, P1))))
            V = np.vstack((V1, V2))
            S0, _ = null_space(V)

        elif decoupled_sys.name == 'AutonomousDecoupledIndexThree':

            # for index-3 decoupled autonomous dae
            # the consistent condition is:
            #            1) x2(0) = A2 * x1(0)
            #            2) x3(0) = A3 * x1(0) + C3 * dot{x2}(0)
            #            3) x4(0) = A4 * x1(0) + C4 * dot{x3}(0) + D4 * dot{x2}(0)
            # where: x1 = P0 * P1 * P2 * x, x2 = P0 * P1 * Q2 * x, x3 = P0 * Q1, x4 = Q0
            # thus, the consistent condition becomes:
            #            1) P0 * P1 * Q2 * S(0) = A2 * P0 * P1 * P2 * S(0)
            #            2) P0 * Q1 * S(0) = (A3 + C3 * A2 * A1) * P0 * P1 * P2 * S(0)
            #            3) Q0 * S(0) = (A4 + C4 (A3 * A1 + C3 * A2 * A1 * A1) + D4 * A2 * A1) * P0 * P1 * P2 * S(0)

            Q0 = decoupled_sys.projectors[0]
            Q1 = decoupled_sys.projectors[1]
            Q2 = decoupled_sys.projectors[2]
            n = Q0.shape[0]
            In = np.eye(n, dtype=float)
            P0 = In - Q0
            P1 = In - Q1
            P2 = In - Q2
            A1 = decoupled_sys.ode_matrix_a
            A2 = decoupled_sys.alg1_matrix_a
            A3 = decoupled_sys.alg2_matrix_a
            C3 = decoupled_sys.alg2_matrix_c
            A4 = decoupled_sys.alg3_matrix_a
            C4 = decoupled_sys.alg3_matrix_c
            D4 = decoupled_sys.alg3_matrix_d
            P0_P1_P2 = np.dot(P0, P1, P2)
            V1 = np.dot(P0, np.dot(P1, Q2)) - np.dot(A2, P0_P1_P2)
            V2 = np.dot(P0, Q1) - np.dot(A3 +
                                         np.dot(C3, np.dot(A2, A1)), P0_P1_P2)

            V30 = A4 + np.dot(C4, np.dot(A3, A1) + \
                                  np.dot(C3, np.dot(A2, np.dot(A1, A1)))) + np.dot(D4, np.dot(A2, A1))
            V3 = Q0 - np.dot(V30, P0_P1_P2)

            V = np.vstack((V1, V2, V3))
            S0, _ = null_space(V)

        else:
            print "\nunknown type of decoupled system"

        runtime = time.time() - start

        return S0, runtime
