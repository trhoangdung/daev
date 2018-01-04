'''
Tests for set module
Dung Tran: 1/2018
'''

from daev.engine.set import InitSet
import numpy as np

def test():
    'test methods'

    init_cond = InitSet()
    min_vec = [0.0, 0.1, 0.2]
    max_vec = [0.0, 0.2, 0.3]

    init_cond.set_params(min_vec, max_vec)
    initSet = init_cond.get_init_reachset()

    print "\ninit Set S(0) = {}".format(initSet.S)
    print "\ninit Set alpha_min = {}".format(initSet.alpha_min_vec)
    print "\ninit Set alpha_max = {}".format(initSet.alpha_max_vec)

    safe_mat = np.array([[0.0, 1.0, 1.0]])
    safe_vec = np.array([0.4])

    print safe_vec.shape
    print safe_mat.shape
    print safe_mat

    status, feas_sol, unsafe_state, unsafe_vec = initSet.check_safety(safe_mat, safe_vec)

    print "\nsafety status: {}".format(status)
    print "\nfeasible solution = {}".format(feas_sol)
    print "\nunsafe state = {}".format(unsafe_state)
    print "\nunsafe vector (safe_mat * unsafe_state) = {}".format(unsafe_vec)

if __name__ == '__main__':
    test()
