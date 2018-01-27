'''
This implements verification/falsification algorithm for dae system
DungTran: Jan/2018
'''

from daev.engine.dae_automaton import AutonomousDaeAutomation
from daev.engine.reachability import ReachSetAssembler
from daev.engine.set import LinearPredicate
import numpy as np
import time


class VerificationResult(object):
    'contain verification result'

    def __init__(self):
        self.totime = None
        self.num_steps = None
        self.solver = None
        self.status = None    # safe/unsafe
        self.unsafe_point = None
        self.unsafe_trace = []    # contain unsafe trace
        # contain unsafe state (trace of all state variables)
        self.unsafe_state_trace = []
        self.runtime = None    # time for verification/falsification
        self.reach_set = None    # contain reachable set
        self.unsafe_set = None   # contain unsafe set


class Verifier(object):
    'Verifier class'

    def __init__(self):
        self.verification_result = None    # contain verification result object

    def check_safety(self, dae_sys, init_set, unsafe_set,
                     totime, num_steps, solver_name):
        'check safety of a dae system'

        assert isinstance(dae_sys, AutonomousDaeAutomation)
        assert isinstance(unsafe_set, LinearPredicate)
        reach_set, decoupling_time, reachset_computation_time = ReachSetAssembler(
        ).reach_autonomous_dae(dae_sys, init_set, totime, num_steps, solver_name)
        start = time.time()
        time_list = np.linspace(0.0, totime, num_steps + 1)
        n = len(reach_set)
        status = 'safe'
        unsafe_trace = []
        unsafe_state_trace = []
        unsafe_point = []
        for i in xrange(0, n):
            rs = reach_set[i]
            status, fes_alpha, _, _ = rs.check_safety(unsafe_set)
            if status == 'unsafe':
                constr_mat = np.dot(unsafe_set.C, rs.S)
                unsafe_vec = np.dot(constr_mat, fes_alpha)
                unsafe_point = (unsafe_vec, time_list[i], fes_alpha)
                break
        if status == 'unsafe':
            # compute unsafe trace
            for i in xrange(0, n):
                rs = reach_set[i]
                constr_mat = np.dot(unsafe_set.C, rs.S)
                unsafe_vec = np.dot(constr_mat, fes_alpha)
                unsafe_state = np.dot(rs.S, fes_alpha)
                unsafe_trace.append(unsafe_vec)
                unsafe_state_trace.append(unsafe_state)

        checking_safety_time = time.time() - start

        ver_res = VerificationResult()
        ver_res.totime = totime
        ver_res.num_steps = num_steps
        ver_res.solver = solver_name
        ver_res.status = status
        ver_res.unsafe_point = unsafe_point
        ver_res.unsafe_trace = unsafe_trace
        ver_res.unsafe_state_trace = unsafe_state_trace
        ver_res.runtime = [
            ('decoupling_time',
             decoupling_time),
            ('reachset_computation_time',
             reachset_computation_time),
            ('checking_safety_time',
             checking_safety_time),
            ('verification_time',
             decoupling_time +
             reachset_computation_time +
             checking_safety_time)]
        ver_res.reach_set = reach_set
        ver_res.unsafe_set = unsafe_set
        self.verification_result = ver_res

        return self.verification_result
