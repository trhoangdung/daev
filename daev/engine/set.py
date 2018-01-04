'''
This module implements set classes and methods
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
    #     alpha satisfy the constraint  alpha_min <= alpha <= alpha_max

    def __init__(self):
        self.S = None    # solution matrix S at time t
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

        assert self.S is not None, 'error: empty set'
        assert isinstance(safe_mat, np.ndarray)
        assert isinstance(safe_vec, np.ndarray), 'error: invalid safety constraint'
        assert safe_mat.shape[0] == safe_vec.shape[0], 'error: inconsistent safe_mat and safe_vec'
        assert safe_mat.shape[1] == self.S.shape[0], 'error: inconsistent safe_mat and self.S'

        constr_mat = np.dot(safe_mat, self.S)
        n = self.S.shape[1]
        c_vec = np.ones(n)
        alpha_bounds = []
        for i in xrange(0, n):
            alpha_bounds.append((self.alpha_min_vec[i, 0], self.alpha_max_vec[i, 0]))
        opt_res = linprog(c=c_vec, A_ub=constr_mat, b_ub=safe_vec, bounds=alpha_bounds)

        if opt_res.status == 2:
            status = 'safe'
            fes_alpha = []
            unsafe_vec = None
            unsafe_state = None
        elif opt_res.status == 0:
            status = 'unsafe'
            fes_alpha = opt_res.x    # feasible alpha values
            unsafe_vec = np.dot(constr_mat, fes_alpha)
            unsafe_state = np.dot(self.S, fes_alpha)
        else:
            print "\noptimization error: iteration limit reached or problem is unbounded"

        return status, fes_alpha, unsafe_state, unsafe_vec

    def multiply(self, direction_matrix):
        'project reachset onto a specific direction'

        assert self.S is not None, 'error: empty set'
        assert isinstance(direction_matrix, np.ndarray)
        assert direction_matrix.shape[1] == self.S.shape[0]

        projected_reach_set = ReachSet()
        projected_S = np.dot(direction_matrix, self.S)
        projected_reach_set.set_params(projected_S, self.alpha_min_vec, self.alpha_max_vec)

        return projected_reach_set

    def add(self, other_reachset):
        'add itself with other reachset'

        assert isinstance(other_reachset, ReachSet)
        assert other_reachset.alpha_min_vec == self.alpha_min_vec, 'error: inconsistency, can not add'
        assert other_reachset.alpha_max_vec == self.alpha_max_vec, 'error: inconsistency, can not add'

        new_reach_set = ReachSet()
        new_S = other_reachset.S + self.S
        new_reach_set.set_params(new_S, self.alpha_min_vec, self.alpha_max_vec)

        return new_reach_set


class InitSet(object):
    'initial set of states and inputs'

    # min_vec[i] <= x[i] <= max_vec[i]

    def __init__(self):
        self.min_vec = None
        self.max_vec = None

    def set_params(self, min_vec, max_vec):
        'set parameters for initial set'

        assert isinstance(min_vec, list), 'error: invalid min vector'
        assert isinstance(max_vec, list), 'error: invalid max vector'
        assert len(max_vec) == len(min_vec), 'error: inconsistent min_vec and max_vec'

        n = len(min_vec)
        for i in xrange(0, n):
            assert isinstance(min_vec[i], float), 'error: min_vec values is not float'
            assert isinstance(max_vec[i], float), 'error: max_vec values is not float'

            if min_vec[i] > max_vec[i]:
                raise ValueError('error: min_vec[{}] = {} > max_vec[{}] = {}'.format(i, min_vec[i], i, max_vec[i]))

        self.min_vec = min_vec
        self.max_vec = max_vec

    def get_init_reachset(self):
        'return optimized initial reach set (star set) from init set'

        # x = c + Sigma_i (alpha[i] * v[i])
        # if x[j] is fixed element, then alpha[j] = 0
        # otimized Star Set removes all alpha[i] = 0

        # assume k is number of uncertain elements
        # the initial set X(0) = S(0) * alpha
        # where:
        #    S(0) = [c[0] v_1 v_2 ... v_k]
        #    alpha = [1 alpha[1] alpha[2] ... alpha[k]]^T
        #    v_i is the basic vector of the ith uncertain element
        #    alpha_min_vec <= alpha <= alpha_max_vec

        assert self.min_vec is not None and self.max_vec is not None, 'error: empty initial set'

        n = len(self.min_vec)
        centre_vec = np.zeros((n, 1), dtype=float)
        In = np.eye(n, dtype=float)
        opt_alpha_min_list = []
        opt_alpha_max_list = []
        opt_vi_list = []
        k = 0    # number of uncertain elements
        for i in xrange(0, n):
            if self.min_vec[i] == self.max_vec[i]:
                centre_vec[i] = self.min_vec[i]
            else:
                centre_vec[i] = (self.min_vec[i] + self.max_vec[i]) / 2
                k = k + 1
                opt_alpha_min_list.append(self.min_vec[i] - centre_vec[i])
                opt_alpha_max_list.append(self.max_vec[i] - centre_vec[i])
                opt_vi_list.append(In[:, i])

        opt_alpha_min = np.zeros((k + 1, 1), dtype=float)
        opt_alpha_max = np.zeros((k + 1, 1), dtype=float)
        S0 = np.zeros((n, k + 1), dtype=float)

        for i in xrange(0, k + 1):
            if i == 0:
                opt_alpha_min[i] = 1.0
                opt_alpha_max[i] = 1.0
                S0[:, i] = centre_vec.reshape(n)
            else:
                opt_alpha_min[i] = opt_alpha_min_list[i - 1]
                opt_alpha_max[i] = opt_alpha_max_list[i - 1]
                S0[:, i] = opt_vi_list[i - 1].reshape(n)

        init_reachset = ReachSet()
        init_reachset.set_params(S0, opt_alpha_min, opt_alpha_max)

        return init_reachset
