'''
This modules implements simulation-based reachability analysis for decoupled dae systems
Dung Tran: Dec/2017
'''

import numpy as np
from scipy.optimize import linprog


class ReachSet(object):
    'class for contain reachable set using Star Set'

    # reference: 1) Simulation-Equivalent Reachability of Large Linear Systems with Inputs, S.Bak, CAV, 2017
    # for convenient safety checking and reduce computation cost, we modify the original star set
    # by neglecting the fixed initial state elements

    # assume k is the number of uncertain elements of the initial set
    # the star-set at time t can be represented by
    #     x(t) = S(t) * alpha
    #     S(t) = [c(t), v_1(t), v_2(t), ..., v_k(t)]
    #     alpha = [1 alpha_1 alpha_2 ... alpha_k]^T
    #     where:
    #     c(t) is the solution at time t of the "central point" of the initial start set S(0)
    #     v_i(t) is the solution at time t of the basic vector corresponding to the ith uncertain element
    #     alpha satisfy the constraint C * alpha <= d

    def __init__(self):
        self.S = []    # solution matrix S at time t
        self.alpha_min_vec = None    # constraint on alpha parameters
        self.alpha_max_vec = None

    def set_params(self, S, alpha_min, alpha_max):
        'set parameters for the Reach Set'

        assert isinstance(S, np.ndarray)
        assert isinstance(alpha_min, np.ndarray) and alpha_min.shape[1] == 1
        assert isinstance(alpha_max, np.ndarray) and alpha_max.shape[1] == 1
        assert alpha_min.shape[0] == S.shape[1] == alpha_max.shape[0]
        self.S = S
        self.alpha_min_vec = alpha_min
        self.alpha_max_vec = alpha_max

    def check_safety(self, safe_mat, safe_vec):
        'checking safety'

        # we want to check : safe_mat * x <= safe_vec is feasible or not
        # if unfeasible, return safe
        # if feasisible, return unsafe and the feasisible solution alpha

        assert self.S != [], 'error: empty set'
        assert isinstance(safe_mat, np.ndarray)
        assert isinstance(safe_vec, np.ndarray) and safe_vec.shape[1] == 1, 'error: invalid safety constraint'
        assert safe_mat.shape[0] == safe_vec.shape[0], 'error: inconsistent safe_mat and safe_vec'
        assert safe_mat.shape[0] == self.S[0].shape[0], 'error: inconsistent safe_mat and self.S'

        constr_mat = np.dot(safe_mat, self.S)
        n = len(self.S)
        c_vec = np.ones(n)

        opt_res = linprog(c=c_vec, A_ub=constr_mat, b_ub=safe_vec, bounds=(self.alpha_min_vec, self.alpha_max_vec))

        if opt_res.status == 2:
            status = 'safe'
            fes_alpha = []
        elif opt_res.status == 0:
            status = 'unsafe'
            fes_alpha = opt_res.x    # feasible alpha values
        else:
            print "\noptimization error: iteration limit reached or problem is unbounded"

        return status, fes_alpha










class ReachAssembler(object):
    'class for computing discrete reachable set of decoupled dae system'

    pass
