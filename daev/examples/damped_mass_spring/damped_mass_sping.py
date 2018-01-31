'''
Damped-mass-spring example run file, total runtime = 3 minutes
Dung Tran: Jan/2018
'''

from daev.daes import index_3_daes
from daev.engine.dae_automaton import DaeAutomation
from daev.engine.decoupling import DecouplingAutonomous
from daev.engine.set import LinearPredicate, ReachSet
from daev.engine.verifier import Verifier
from daev.engine.plot import Plot
from daev.engine.printer import spaceex_printer
from scipy.sparse import csc_matrix
import numpy as np


def get_benchmark(N):
    'get benchmark matrices'

    E, A, B, C = index_3_daes().damped_mass_spring(N)
    print "\n########################################################"
    print "\nDAMPED-MASS-SPRING:"
    print "\ndimensions: {}".format(E.shape[0])
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E.todense(), A.todense(), B.todense(), C.todense())

    return E, A, B, C


def construct_dae_automaton(E, A, B, C):
    'create dae automaton for the benchmark'

    dae_sys = DaeAutomation()
    dae_sys.set_dynamics(E, A, B, C)
    return dae_sys


def convert_to_auto_dae(dae_sys):
    'convert dae system to autonomous dae automaton'

    u_mat = np.array([-0.08])    # user-defined input, a constant input
    dae_auto = dae_sys.convert_to_autonomous_dae(csc_matrix(u_mat))
    print "\ndae_auto matrix_e = {}".format(dae_auto.matrix_e.todense())
    print "\nrank of new E = {}".format(np.linalg.matrix_rank(dae_auto.matrix_e.todense()))
    print "\ndae_auto matrix_a = {}".format(dae_auto.matrix_a.todense())
    print "\ndae_auto matrix_c = {}".format(dae_auto.matrix_c.todense())

    return dae_auto


def decouple_auto_dae(dae_auto):
    'decoupling autonomous dae system'

    decoupled_dae, status = DecouplingAutonomous().get_decoupled_system(dae_auto)

    print "\ndecoupling status = {}".format(status)
    print "\ndecoupled dae_auto: N1 = {}".format(decoupled_dae.N1)
    print "\nnorm of N1 = {}".format(np.linalg.norm(decoupled_dae.N1))
    print "\ndecoupled dae_auto: N2 = {}".format(decoupled_dae.N2)
    print "\nnorm of N2 = {}".format(np.linalg.norm(decoupled_dae.N2))
    print "\ndecoupled dae_auto: N3 = {}".format(decoupled_dae.N3)
    print "\nnorm of N3 = {}".format(np.linalg.norm(decoupled_dae.N3))
    print "\ndecoupled dae_auto: L3 = {}".format(decoupled_dae.L3)
    print "\nnorm of L3 = {}".format(np.linalg.norm(decoupled_dae.L3))
    print "\ndecoupled dae_auto: N4 = {}".format(decoupled_dae.N4)
    print "\nnorm of N4 = {}".format(np.linalg.norm(decoupled_dae.N4))
    print "\ndecoupled dae_auto: L4 = {}".format(decoupled_dae.L4)
    print "\nnorm of L4 = {}".format(np.linalg.norm(decoupled_dae.L4))
    print "\ndecoupled dae_auto: Z4 = {}".format(decoupled_dae.Z4)
    print "\nnorm of Z4 = {}".format(np.linalg.norm(decoupled_dae.Z4))

    return decoupled_dae


def generate_consistent_basic_matrix(decoupled_dae):
    'generate an consistent basic_matrix for initial condition'

    basic_matrix = decoupled_dae.generate_consistent_init_set_basic_matrix()
    print "\nconsistent basic matrix: \n{}".format(basic_matrix)
    print "\nnorm of basic matrix = {}".format(np.linalg.norm(basic_matrix))
    print "\nbasic matrix shape = {}".format(basic_matrix.shape)

    return basic_matrix


def construct_init_set(basic_matrix):
    'construct linear predicate for initial set'

    init_set_basic_matrix = basic_matrix[:, 0:2]
    print "\ninit_set_basic_matrix shape = {}".format(init_set_basic_matrix.shape)
    alpha_min = np.array([[1.0], [2.0]])
    alpha_max = np.array([[1.2], [2.2]])

    print "\ninitial set basic matrix: \n{}".format(init_set_basic_matrix)
    print "\ninitial set alpha min: \n{}".format(alpha_min)
    print "\ninitial set alpha max: \n{}".format(alpha_max)

    init_set = ReachSet()
    init_set.set_basic_matrix(init_set_basic_matrix)
    init_set.set_alpha_min_max(alpha_min, alpha_max)

    return init_set


def construct_unsafe_set(dae_auto):
    'construct unsafe set'

    # unsafe set: p_c >= 1.0 and v_c >= 1.5, the middle mass position
    c_mat = dae_auto.matrix_c.todense()
    C = -c_mat
    d = np.array([[-1.0], [-1.5]])
    print "\nunsafe_set 1:  matrix C = {}".format(C)
    print "\nunsafe_set 1:  vector d = {}".format(d)
    unsafe_set1 = LinearPredicate(C, d)

    C = -c_mat[1]    # v_c <= -0.2
    d = np.array([[0.2]])
    print "\nunsafe_set 2:  matrix C = {}".format(C)
    print "\nunsafe_set 2:  vector d = {}".format(d)
    unsafe_set2 = LinearPredicate(C, d)

    unsafe_set = [unsafe_set1, unsafe_set2]    # list of unsafe set

    return unsafe_set


def verify_safety(dae_auto, init_set, unsafe_set, totime, num_steps, solver_name):
    'verify the safety of the system'

    n = len(unsafe_set)
    ver_res = []
    for i in xrange(0, n):
        us = unsafe_set[i]
        vr = Verifier().check_safety(dae_auto, init_set, us, totime, num_steps, solver_name, 'verification_result_case_{}'.format(i))
        ver_res.append(vr)

    return ver_res


def main():
    'main function'

    print "\n############################################################"
    print "\nVERIFY/FALSIFY SAFETY PROPERTY OF A DAMPED-MASS-SPRING SYSTEM"
    E, A, B, C = get_benchmark(5)
    dae_sys = construct_dae_automaton(E, A, B, C)
    dae_auto = convert_to_auto_dae(dae_sys)
    decoupled_dae = decouple_auto_dae(dae_auto)
    basic_matrix = generate_consistent_basic_matrix(decoupled_dae)
    init_set = construct_init_set(basic_matrix)

    totime = 100.0
    num_steps = 1000
    solver_names = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']    # similar to ode45 mathlab

    # print spaceex model
    spaceex_printer(decoupled_dae, init_set, totime, 0.1, 'damped_mass_spring')

    unsafe_set = construct_unsafe_set(dae_auto)
    veri_res = verify_safety(dae_auto, init_set, unsafe_set, totime, num_steps, solver_names[3])
    for vr in veri_res:
        if vr.status == 'unsafe':
            Plot().plot_unsafe_trace(vr)

    # Plot().plot_state_reachset_vs_time(vr)    # plot state reach set


if __name__ == '__main__':

    main()
