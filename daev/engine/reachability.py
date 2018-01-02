'''
This modules implements simulation-based reachability analysis for decoupled dae systems
Dung Tran: Dec/2017
'''

import time
import numpy as np
from daev.engine.set import ReachSet, InitSet
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
            sol, _ = ReachSetAssembler().ode_sim(matrix_a, init_vec, totime, num_steps, solver_name)
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


def test_ode_sim():
    'test methods'

    A = np.array([[0.0, -1.0], [-1.0, 0.0]])
    init_vec = np.array([1, -0.25])
    final_time = 2.0
    num_steps = 10

    solution, runtime = ReachSetAssembler().ode_sim(A, init_vec, final_time, num_steps, 'dopri5')

    print "\nsolution = {}".format(solution)
    print "\nruntime = {}".format(runtime)


def test_reach_autonomous_ode():
    'test reach_autonomous_ode method'

    A = np.array([[0.0, -1.0], [-1.0, 0.0]])
    min_vec = [0.0, 0.1]
    max_vec = [0.0, 0.2]
    init_cond = InitSet()
    init_cond.set_params(min_vec, max_vec)
    initSet = init_cond.get_init_reachset()
    totime = 2.0
    num_steps = 20
    solver_name = 'dopri5'
    reach_set_list, runtime = ReachSetAssembler.reach_autonomous_ode(A, initSet, totime, num_steps, solver_name)
    print "\nreachable set computation runtime = {}".format(runtime)
    print "\nreachable set list = {}".format(reach_set_list)

if __name__ == '__main__':
    #test_ode_sim()
    test_reach_autonomous_ode()