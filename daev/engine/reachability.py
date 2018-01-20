'''
This modules implements simulation-based reachability analysis for decoupled dae systems
Dung Tran: Dec/2017
'''

import time
import numpy as np
from daev.engine.set import ReachSet
from daev.engine.decoupling import AutonomousDaeAutomation, DecouplingAutonomous
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
            init_vec = np.asarray(matrix_S[:, j])
            init_vec = init_vec.reshape(n,)

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

            reach_set.set_basic_matrix(s_mat)
            if alpha_min is not None and alpha_max is not None:
                reach_set.set_alpha_min_max(alpha_min, alpha_max)

            if init_reachset.predicate is not None:
                reach_set.set_predicate(init_reachset.predicate)

            reach_set_list.append(reach_set)

        runtime = time.time() - start

        return reach_set_list, runtime

    @staticmethod
    def reach_autonomous_dae(dae_sys, init_reachset, totime, num_steps, solver_name):
        'compute reachable set for decoupled dae'

        start = time.time()
        assert isinstance(dae_sys, AutonomousDaeAutomation)
        assert isinstance(init_reachset, ReachSet)
        decoupled_sys, status = DecouplingAutonomous().get_decoupled_system(dae_sys)
        if status == 'error':
            raise ValueError('error in decoupling')

        x_reach_set_list = []
        consistency = decoupled_sys.check_consistency(init_reachset)

        if consistency:
            x1_init_reachset = init_reachset.multiply(decoupled_sys.x1_init_set_projector)
            x1_reach_set_list, _ = ReachSetAssembler.reach_autonomous_ode(decoupled_sys.N1, x1_init_reachset, totime, num_steps, solver_name)
            n = len(x1_reach_set_list)
            for i in xrange(0, n):
                x_reach_set_list.append(x1_reach_set_list[i].multiply(decoupled_sys.reach_set_projector))

        else:
            print "\nerror: inconsistent initial condition"

        runtime = time.time() - start

        return x_reach_set_list, runtime
