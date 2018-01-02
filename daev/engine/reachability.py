'''
This modules implements simulation-based reachability analysis for decoupled dae systems
Dung Tran: Dec/2017
'''

import time
import numpy as np
from daev.engine.set import ReachSet
from scipy.integrate import ode
from scipy.sparse import csc_matrix

class ReachSetAssembler(object):
    'implements rechable set computation for odes, daes'

    @staticmethod
    def ode_sim(matrix_a, init_vec, totime, num_steps, solver_name):
        'compute simulation trace of an ode'
        # solvers can be selected from the list = ['vode', 'zvode',  'Isoda', 'dopri5', 'dop853']

        start = time.time()
        assert isinstance(matrix_a, np.ndarray), 'error: matrix_a is not ndarray'
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
    def reach_autonomous_ode(matrix_a, init_reachset, step, num_steps, solvers):
        'compute reachable set of automnous linear ode: \dot{x} = Ax'

        # compute reachable set using simulation
        # solvers can be selected from the list = ['vode', 'zvode',  'Isoda', 'dopri5', 'dop853']

        assert isinstance(matrix_a, np.ndarray) and matrix_a.shape[0] == matrix_a.shape[1], 'error: invalid matrix a'
        assert isinstance(init_reachset, ReachSet)
        assert matrix_a.shape[0] == init_reachset.S.shape[0], 'error: inconsistent matrix_a and initial reach set'
        assert isinstance(step, float) and step > 0, 'error: invalid step size'
        assert isinstance(num_steps, int) and num_steps >= 0, 'error: invalid number of steps'

        pass


def test():
    'test methods'

    A = np.array([[0.0, -1.0], [-1.0, 0.0]])
    init_vec = np.array([1, -0.25])
    final_time = 2.0
    num_steps = 10

    solution, runtime = ReachSetAssembler().ode_sim(A, init_vec, final_time, num_steps, 'dopri5')

    print "\nsolution = {}".format(solution)

    print "\nruntime = {}".format(runtime)


if __name__ == '__main__':
    test()
