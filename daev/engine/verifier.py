'''
This implements verification/falsification algorithm for dae system
DungTran: Jan/2018
'''

from daev.engine.dae_automaton import AutonomousDaeAutomation
from daev.engine.reachability import ReachSetAssembler


class VerificationResult(object):
    'contain verification result'

    def __init__(self):
        self.status = None    # safe/unsafe
        self.unsafe_trace = []    # contain unsafe trace
        self.runtime = None    # time for verification/falsification


class Verifier(object):
    'Verifier class'

    def __init__(self):
        self.verification_result = None    # contain verification result object

    def check_safety(self, dae_sys, init_set, unsafe_set, totime, num_steps, solver_name):
        'check safety of a dae system'

        assert isinstance(dae_sys, AutonomousDaeAutomation)
        reach_set, _ = ReachSetAssembler().reach_autonomous_dae(dae_sys, init_reachset, totime, num_steps, solver_name)

        n = len(reach_set)
