'''
Stokes-equation example run file
Dung Tran: Jan/2018
'''

from daev.daes import index_2_daes
from daev.engine.dae_automaton import DaeAutomation
from daev.engine.decoupling import DecouplingAutonomous
from daev.engine.set import LinearPredicate, ReachSet
from daev.engine.reachability import ReachSetAssembler
from daev.engine.verifier import Verifier
from scipy.sparse import csc_matrix
import numpy as np
import matplotlib.pyplot as plt


def get_benchmark(L, N):
    'get benchmark matrices'

    # STOKES-EQUATION
    E, A, B, C = index_2_daes().stoke_equation_2d(L, N)
    print "\n########################################################"
    print "\nSTOKES-EQUATION:"
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

    u_mat = np.array([0])    # user-defined input, a constant input
    dae_auto = dae_sys.convert_to_autonomous_dae(csc_matrix(u_mat))
    print "\ndae_auto matrix_e = {}".format(dae_auto.matrix_e.todense())
    print "\nrank of new E = {}".format(np.linalg.matrix_rank(dae_auto.matrix_e.todense()))
    print "\ndae_auto matrix_a = {}".format(dae_auto.matrix_a.todense())
    print "\ndae_auto matrix_c = {}".format(dae_auto.matrix_c.todense())

    return dae_auto


def decouple_auto_dae(dae_auto):
    'decoupling autonomous dae system'

    decoupled_dae, status = DecouplingAutonomous().get_decoupled_system(dae_auto)

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
    alpha_max = np.array([[1.5], [2.5]])

    print "\ninitial set basic matrix: \n{}".format(init_set_basic_matrix)
    print "\ninitial set alpha min: \n{}".format(alpha_min)
    print "\ninitial set alpha max: \n{}".format(alpha_max)

    init_set = ReachSet()
    init_set.set_basic_matrix(init_set_basic_matrix)
    init_set.set_alpha_min_max(alpha_min, alpha_max)

    return init_set


def construct_unsafe_set1(dae_auto):
    'construct unsafe set'

    # unsafe set: - vcx - vcy <=
    c_mat = dae_auto.matrix_c.todense()
    C1 = c_mat[0]
    C2 = c_mat[1]
    C = -C1 - C2
    d = np.array([[0.04]])
    print "\nunsafe matrix C = {}".format(C)
    print "\nunsafe vector d = {}".format(d)
    unsafe_set = LinearPredicate(C, d)

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


def get_verification_time():
    'get verification time versus system dimension'

    totime = 0.4
    num_steps = 50
    solver_names = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']    # similar to ode45 mathlab

    # num_meshpoints = [2, 4, 6, 8, 10, 12]
    num_meshpoints = [5, 10, 15, 20, 25, 30]

    verification_time = []

    for num in num_meshpoints:
        E, A, B, C = get_benchmark(1.0, num)
        dae_sys = construct_dae_automaton(E, A, B, C)
        dae_auto = convert_to_auto_dae(dae_sys)
        decoupled_dae = decouple_auto_dae(dae_auto)
        basic_matrix = generate_consistent_basic_matrix(decoupled_dae)
        init_set = construct_init_set(basic_matrix)
        unsafe_set = construct_unsafe_set1(dae_auto)
        veri_res = Verifier().check_safety(dae_auto, init_set, unsafe_set, totime, num_steps, solver_names[3], 'VR_for_dimension_{}'.format(E.shape[0]))
        verification_time.append(('dimension = {}'.format(dae_auto.matrix_e.shape[0]), veri_res.runtime))

    data_file = open('verification_time_vs_dimensions.dat', 'w')
    data_file.write("\n    VERIFICATION TIME\n")

    for i in xrange(len(verification_time)):
        data_file.write("\n{}\n".format(verification_time[i]))
    data_file.close()

    print "\nverification time:"
    for i in xrange(len(verification_time)):
        print "\n{}".format(verification_time[i])

    return verification_time


def get_ode_solvers_time():
    'get reachable set computation time with different ode solvers'

    totime = 0.4
    num_steps = 50
    solver_names = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']

    # num_meshpoints = [2, 3, 4, 5, 6]    # for testing
    num_meshpoints = [5, 10, 15, 20, 25]
    reach_times = []
    for num in num_meshpoints:
        E, A, B, C = get_benchmark(1.0, num)
        dae_sys = construct_dae_automaton(E, A, B, C)
        dae_auto = convert_to_auto_dae(dae_sys)
        decoupled_dae = decouple_auto_dae(dae_auto)
        basic_matrix = generate_consistent_basic_matrix(decoupled_dae)
        init_set = construct_init_set(basic_matrix)

        solver_times = []
        for solver in solver_names:
            _, _, reachset_computation_time = ReachSetAssembler.reach_autonomous_dae(dae_auto, init_set, totime, num_steps, solver)
            print "\n solver = {} -> reachable set computation time = {}".format(solver, reachset_computation_time)
            solver_times.append((('solver', solver), ('reachset_compuation_time', reachset_computation_time)))

        reach_times.append((('dimensions', dae_auto.matrix_a.shape[0]), solver_times))

    data_file = open('reach_time_vs_solvers.dat', 'w')
    data_file.write("\n    REACHABLE SET COMPUTATION TIME VERSUS SOLVERS\n")

    for i in xrange(len(reach_times)):
        data_file.write("\n{}\n".format(reach_times[i]))
    data_file.close()

    return reach_times


def plot_reach_times_vs_solvers(reach_times):
    'plot figure to compare reachable set computation time for different solvers'

    dimensions = []
    vode_times = []
    zvode_times = []
    lsoda_times = []
    dopri5_times = []
    dop853_times = []

    for i in xrange(0, len(reach_times)):
        rt = reach_times[i]
        dim = rt[0]
        rt1 = rt[1]
        dimensions.append(dim[1])
        vode = rt1[0]
        vode1 = vode[1]
        zvode = rt1[1]
        zvode1 = zvode[1]
        lsoda = rt1[2]
        lsoda1 = lsoda[1]
        dopri5 = rt1[3]
        dopri51 = dopri5[1]
        dop853 = rt1[4]
        dop8531 = dop853[1]
        vode_times.append(vode1[1])
        zvode_times.append(zvode1[1])
        lsoda_times.append(lsoda1[1])
        dopri5_times.append(dopri51[1])
        dop853_times.append(dop8531[1])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.plot(dimensions, vode_times, 'x', ls='-', linewidth=1.5)
    ax1.plot(dimensions, zvode_times, '*', ls='-', linewidth=1.5)
    ax1.plot(dimensions, lsoda_times, 'o', ls='-', linewidth=1.5)
    ax1.plot(dimensions, dopri5_times, '^', ls='-', linewidth=1.5)
    ax1.plot(dimensions, dop853_times, 'v', ls='-', linewidth=1.5)

    ax1.legend(['vode', 'zvode', 'lsoda', 'dopri5', 'dop853'])
    ax1.set_ylim(0, 600)
    ax1.set_xlim(min(dimensions) - 50, max(dimensions) + 50)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$n$ (System dimension)', fontsize=20)
    plt.ylabel(r'Computation time (seonds)', fontsize=20)
    fig1.suptitle('Reachable set computation time vs. solvers', fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    fig1.savefig('reach_time_vs_solvers.pdf')


def get_reach_time_vs_num_basic_vectors():
    'reachable set computation time vs number of basic vectors. i.e. number of column of init_set_basic_matrix'

    E, A, B, C = get_benchmark(1.0, 10)
    dae_sys = construct_dae_automaton(E, A, B, C)
    dae_auto = convert_to_auto_dae(dae_sys)
    decoupled_dae = decouple_auto_dae(dae_auto)
    basic_matrix = generate_consistent_basic_matrix(decoupled_dae)

    n = basic_matrix.shape[1]
    num_basic_vecs = np.arange(2, n, step=2)
    print "\nnumber of basic vectors = {}".format(num_basic_vecs)

    totime = 0.4
    num_steps = 1000
    solver = 'dopri5'    # similar to ode45 mathlab
    reach_times = []

    for k in num_basic_vecs:
        alpha_min = np.zeros((k, 1))
        alpha_max = np.ones((k, 1))

        init_set_basic_matrix = basic_matrix[:, 0:k]
        print "\ninit_set_basic_matrix shape = {}".format(init_set_basic_matrix.shape)
        print "\ninitial set basic matrix: \n{}".format(init_set_basic_matrix)
        print "\ninitial set alpha min: \n{}".format(alpha_min)
        print "\ninitial set alpha max: \n{}".format(alpha_max)

        init_set = ReachSet()
        init_set.set_basic_matrix(init_set_basic_matrix)
        init_set.set_alpha_min_max(alpha_min, alpha_max)

        _, _, reachset_computation_time = ReachSetAssembler.reach_autonomous_dae(dae_auto, init_set, totime, num_steps, solver)

        reach_times.append(('solver={}, dimension={}, num_steps={}, totime={}, num_basic_vectors = {}'.format(solver, dae_auto.matrix_a.shape[0], num_steps, totime, k), reachset_computation_time))
        print "\nsolver={}, dimension={}, num_steps={}, totime={}, num_basic_vectrors = {} -> reach_time = {}".format(solver, dae_auto.matrix_a.shape[0], num_steps, totime, k, reachset_computation_time)

    data_file = open('reach_time_vs_num_basic_vectors.dat', 'w')
    data_file.write("\nREACHABLE SET COMPUTATION TIME VERSUS NUMBER OF BASIC VECTORS OF THE BASIC MATRIX\n")

    for i in xrange(len(reach_times)):
        data_file.write("\n{}\n".format(reach_times[i]))
    data_file.close()

    return reach_times


def verify_large_stokes_equation(N):
    'verify a large stokes equation'

    E, A, B, C = get_benchmark(1.0, N)
    dae_sys = construct_dae_automaton(E, A, B, C)
    dae_auto = convert_to_auto_dae(dae_sys)
    decoupled_dae = decouple_auto_dae(dae_auto)
    basic_matrix = generate_consistent_basic_matrix(decoupled_dae)
    init_set = construct_init_set(basic_matrix)

    totime = 0.4
    num_steps = 100
    solver_names = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']    # similar to ode45 mathlab

    # unsafe set: - vcx - vcy <=
    c_mat = dae_auto.matrix_c.todense()
    C1 = c_mat[0]
    C2 = c_mat[1]
    C = -C1 - C2
    d = np.array([[0.04]])    # unsafe
    print "\nunsafe_set 1:  matrix C = {}".format(C)
    print "\nunsafe_set 1:  vector d = {}".format(d)
    unsafe_set1 = LinearPredicate(C, d)
    C = -C1
    d = np.array([[-0.2]])    # safe vcx >= 0.2
    print "\nunsafe_set 2:  matrix C = {}".format(C)
    print "\nunsafe_set 2:  vector d = {}".format(d)
    unsafe_set2 = LinearPredicate(C, d)
    unsafe_set = [unsafe_set1, unsafe_set2]    # list of unsafe set

    # verify the safety of the system'
    n = len(unsafe_set)
    ver_res = []
    for i in xrange(0, n):
        us = unsafe_set[i]
        vr = Verifier().check_safety(dae_auto, init_set, us, totime, num_steps, solver_names[3], 'vr_dimension_{}_case_{}'.format(E.shape[0], i))
        ver_res.append(vr)

    return ver_res


def main():
    'main function'

    print "\n##############TIME COMPLEXITY ANALYSIS#################"
    print "\ntaking about 4 hour to complete...."
    print "\nplease select one in the following options to go, you can select all"

    # verification_time = get_verification_time()    # verification time versus dimension this takes about 45 minutes
    # solver_times = get_ode_solvers_time()    # this takes about 45 minutes
    # plot_reach_times_vs_solvers(solver_times)

    # get_reach_time_vs_num_basic_vectors()    # this takes about 3 mintues

    # verify large stokes-equation
    veri_res = verify_large_stokes_equation(40)    # this take 60 minutes
    # veri_res = verify_large_stokes_equation(60)

if __name__ == '__main__':

    main()
