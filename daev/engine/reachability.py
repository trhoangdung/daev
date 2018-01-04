'''
This modules implements simulation-based reachability analysis for decoupled dae systems
Dung Tran: Dec/2017
'''

import time
import numpy as np
from daev.engine.set import ReachSet
from daev.engine.decoupling import AutonomousDecoupledIndexOne
from daev.engine.dae_automaton import DaeAutomation
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
    def reach_autonomous_dae_index_1(
            decoupled_sys, init_reachset, totime, num_steps, solver_name):
        'compute reachable set of index-1 autonomous dae system'

        start = time.time()
        assert isinstance(
            decoupled_sys, AutonomousDecoupledIndexOne), 'error: decoupled system is not index 1 autonomous dae'
        assert isinstance(
            init_reachset, ReachSet), 'error: init_reach set is not a ReachSet type'
        assert isinstance(
            totime, float) and totime > 0, 'error: invalid final time'
        assert isinstance(
            num_steps, int) and num_steps > 0, 'error: invalid number of steps'

        if solver_name != 'vode' and solver_name != 'zvode' and solver_name != 'Isoda' and solver_name != 'dopri5' and solver_name != 'dop853':
            raise ValueError('error: invalid solver name')

        A1 = decoupled_sys.ode_matrix_a
        A2 = decoupled_sys.alg_matrix_a

        # check consistent condition: x2(0) = A2 * x1(0) or Q * x(0) = A2 * P *
        # x(0)
        S0 = init_reachset.S

        print "\nA2 = {}".format(A2)
        print "\nS0 = {}".format(S0)
        print "\nA2 * S0 = {}".format(np.dot(A2, S0))

        if np.linalg.norm(np.dot(A2, S0)) > 1e-6:
            raise ValueError('error: inconsistent initial condition')
        else:
            x1_reach_set_list, _ = ReachSetAssembler().reach_autonomous_ode(
                A1, init_reachset, totime, num_steps, solver_name)
            x2_reach_set_list = []
            x_reach_set_list = []    # x = x1 + x2
            n = len(x1_reach_set_list)
            for i in xrange(0, n):
                x2_reach_set_list.append(x1_reach_set_list[i].multiply(A2))
                x_reach_set_list.append(
                    x1_reach_set_list[i].add(
                        x2_reach_set_list[i]))

        runtime = time.time() - start

        return x_reach_set_list, runtime

    @staticmethod
    def generate_consistent_init_condition(decoupled_sys):
        'generate a space for consistent initial condition'

        # initset is X(0) = S(0) * alpha
        # for index-1 decoupled autonomous dae
        # consistent condition is : x2(0) = A2 * x1(0), x2 = Q * x, x1 = P * x
        # thus, we need: QS(0) = A2 * P * S(0) or S(0) is null-space of (Q - A2 * P)


        # for index-2 decoupled autonomous dae
        # the consistent condition is: x2(0) = A2 * x1(0), x3(0) = A3 * x1(0) + C3 * dot{x2}(0)
        # to make it simple we further assume that dot{x}(0) = 0
        # thus the consistent condition is:
        #            1) Q1*S(0) = A2 * P0 * P1 * S(0)
        #            2) Q0*S(0) = A3 * P0 * P1 * S(0)
        # or S(0) is a null space of V, where V = [Q1 - A2 * P1; Q0 - A3 * P1]


        # for index-3 decoupled autonomous dae
        # the consistent condition is:
        #            1) x2(0) = A2 * x1(0)
        #            2) x3(0) = A3 * x1(0) + C3 * dot{x2}(0)
        #            3) x4(0) = A4 * x1(0) + C4 * dot{x3}(0) + D4 * dot{x2}(0)
        # to make it simple we futher assume that: dot{x}(0) = 0
        # thus, the consistent condition becomes:
        #            1) P0 * P1 * Q2 * S(0) = A2 * P0 * P1 * P2 * S(0)
        #            2)
