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
        self.sys_dim = None
        self.num_inputs = None    # use to plot unsafe trace
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
                     totime, num_steps, solver_name, file_name):
        'check safety of a dae system'

        # file name is used to save the veification result
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
                fes_alpha = fes_alpha.reshape(fes_alpha.shape[0], 1)
                unsafe_vec = np.dot(constr_mat, fes_alpha)
                unsafe_point = (unsafe_vec, time_list[i], fes_alpha)
                break
        if status == 'unsafe':
            # compute unsafe trace
            for i in xrange(0, n):
                rs = reach_set[i]
                constr_mat = np.dot(unsafe_set.C, rs.S)
                unsafe_vec = np.dot(constr_mat, fes_alpha)
                print "\nshape of fes_alpha = {}".format(fes_alpha.shape)
                print "\nunsafe vec = {}".format(unsafe_vec)
                unsafe_state = np.dot(rs.S, fes_alpha)
                unsafe_trace.append(unsafe_vec)
                unsafe_state_trace.append(unsafe_state)

        checking_safety_time = time.time() - start

        ver_res = VerificationResult()
        ver_res.sys_dim = dae_sys.matrix_a.shape[0]
        ver_res.num_inputs = dae_sys.num_inputs
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

        # print and save result
        print "\n##########VERIFICATION RESULT###########"
        print "\nsystem dimension: {}".format(ver_res.sys_dim)
        print "\nunsafe set:"
        print "\nunsafe matrix = \n {}".format(ver_res.unsafe_set.C)
        print "\nunsafe vector = \n {}".format(ver_res.unsafe_set.d)
        print "\nsafety status = {}".format(ver_res.status)
        print "\nruntime = {}".format(ver_res.runtime)
        if ver_res.status == 'unsafe':
            print "\nunsafe_point: output = {}, t = {} seconds, fes_alpha = {}".format(ver_res.unsafe_point[0], ver_res.unsafe_point[1], ver_res.unsafe_point[2])

        assert isinstance(file_name, str)
        data_file = open(file_name, 'w')
        data_file.write("\nVERIFICATION RESULT\n")
        data_file.write("\nSystem dimension: {}\n".format(ver_res.sys_dim))
        data_file.write("\nUnsafe set:\n")
        data_file.write("\nUnsafe matrix : C = \n {}\n".format(ver_res.unsafe_set.C))
        data_file.write("\nUnsafe vector : d = \n {}\n".format(ver_res.unsafe_set.d))
        data_file.write("\nToTime: {}\n".format(ver_res.totime))
        data_file.write("\nNumber of steps: {}\n".format(ver_res.num_steps))
        data_file.write("\nSafety Status: {}\n".format(ver_res.status))
        data_file.write("\nUnsafe Point: {}\n".format(ver_res.unsafe_point))
        data_file.write("\nRuntime: {}\n".format(ver_res.runtime))
        data_file.close()

        return self.verification_result
