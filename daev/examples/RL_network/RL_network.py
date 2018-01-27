'''
RL network example run file
Dung Tran: Jan/2018
'''

from daev.daes import index_2_daes
from daev.engine.dae_automaton import DaeAutomation
from daev.engine.decoupling import DecouplingAutonomous
from daev.engine.set import LinearPredicate, ReachSet
from daev.engine.reachability import ReachSetAssembler
from daev.engine.verifier import Verifier
from daev.engine.plot import Plot
from daev.engine.projectors import admissible_projectors
from scipy.sparse import csc_matrix
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def get_benchmark():
    'get benchmark matrices'

    E, A, B, C = index_2_daes().RL_network(1.0, 2.0)
    print "\n########################################################"
    print "\nRL NETWORK:"
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

    u_mat = np.array([-2])    # user-defined input
    dae_auto = dae_sys.convert_to_autonomous_dae(csc_matrix(u_mat))
    print "\ndae_auto matrix_e = {}".format(dae_auto.matrix_e.todense())
    print "\nrank of new E = {}".format(np.linalg.matrix_rank(dae_auto.matrix_e.todense()))
    print "\ndae_auto matrix_a = {}".format(dae_auto.matrix_a.todense())
    print "\ndae_auto matrix_c = {}".format(dae_auto.matrix_c.todense())

    return dae_auto


def get_admissible_projectors(dae_auto):
    'get admissible projectors for the autonomous dae system'

    adm_projs, _, _, _ = admissible_projectors(dae_auto.matrix_e.todense(), dae_auto.matrix_a.todense())
    print "\nadmissible projectors:"
    print "\nadm_projs = {}".format(adm_projs)
    print "\nQ0 = {}".format(adm_projs[0])
    print "\nQ1 = {}".format(adm_projs[1])

    return adm_projs


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

    init_set_basic_matrix = basic_matrix[:, 0:1]
    print "\ninit_set_basic_matrix shape = {}".format(init_set_basic_matrix.shape)

    alpha_min = np.array([[0.5]])
    alpha_max = np.array([[0.8]])

    print "\ninitial set basic matrix: \n{}".format(init_set_basic_matrix)
    print "\ninitial set alpha min: \n{}".format(alpha_min)
    print "\ninitial set alpha max: \n{}".format(alpha_max)

    init_set = ReachSet()
    init_set.set_basic_matrix(init_set_basic_matrix)
    init_set.set_alpha_min_max(alpha_min, alpha_max)

    return init_set


def construct_unsafe_set(dae_auto):
    'construct unsafe set'

    # unsafe set: x_1 + x_2 + x_3 >= 0.1
    C1 = dae_auto.matrix_c.todense()
    C = np.zeros((1, C1.shape[1]))
    C[0, 1] = -1
    d = np.array([[-0.1]])
    print "\nunsafe matrix C = {}".format(C)
    print "\nunsafe vector d = {}".format(d)
    unsafe_set = LinearPredicate(C, d)

    return unsafe_set


def compute_reachable_set(dae_auto, init_set, totime, num_steps, solver_name):
    'compute reachable set'

    reachset, decoupling_time, reachset_computation_time = ReachSetAssembler.reach_autonomous_dae(dae_auto, init_set, totime, num_steps, solver_name)
    print "\nlength of reachset = {}".format(len(reachset))
    print "\ndecoupling time = {}".format(decoupling_time)
    print "\nruntime of computing reachable set = {}".format(reachset_computation_time)

    for i in xrange(0, len(reachset)):
        print "\nreachset_basic_matrix[{}] = \n{}".format(i, reachset[i].S)

    return reachset


def get_line_set(reachset, direction_matrix):
    'get list of line set to plot the reachable set'

    list_of_line_set_list = []
    print "\ndirection_matrix = {}".format(direction_matrix)
    for i in xrange(0, len(reachset)):
        line_set = reachset[i].get_line_set(direction_matrix)
        list_of_line_set_list.append(line_set)
        print "\nline_set_list[{}] = {}".format(i, line_set)

    return list_of_line_set_list


def plot_vline_set(list_of_line_set_list, totime, num_steps):
    'plot reach set of individual outputs'

    n = len(list_of_line_set_list)    # number of line_set list
    m = len(list_of_line_set_list[0])    # number of outputs
    time_list = np.linspace(0.0, totime, num_steps + 1)
    print "\ntime_list = {}".format(time_list)

    colors = ['b', 'g', 'maroon', 'c']
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    pl1 = Plot()

    for i in xrange(0, m - 1):
        line_set_output_i = []
        for j in xrange(0, n):
            line_set_list = list_of_line_set_list[j]
            line_set_output_i.append(line_set_list[i])
            print "\noutput_{} at step {}: min = {}, max = {}".format(i, j, line_set_list[i].xmin, line_set_list[i].xmax)

        ax1 = pl1.plot_vlines(ax1, time_list.tolist(), line_set_output_i, colors=colors[i], linestyles='solid')

    ax1.legend([r'$x_{1}(t) + x_2(t) + x_3(t)$'], fontsize=20)
    ax1.set_ylim(-1.0, 0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel(r'$x_{1}(t) + x_2(t) + x_3(t)$', fontsize=20)
    fig1.suptitle('Output reachable set', fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig1.savefig('Output_reachset.pdf')
    plt.show()


def verify_safety(dae_auto, init_set, unsafe_set, totime, num_steps, solver_name):
    'verify the safety of the system'

    veri_result = Verifier().check_safety(dae_auto, init_set, unsafe_set, totime, num_steps, solver_name)
    print "\nsafety status = {}".format(veri_result.status)
    print "\nruntime = {}".format(veri_result.runtime)
    if veri_result.status == 'unsafe':
        print "\nunsafe_point: output = {}, t = {} seconds, fes_alpha = {}".format(veri_result.unsafe_point[0], veri_result.unsafe_point[1], veri_result.unsafe_point[2])

    data_file = open('verification_result.dat', 'w')
    data_file.write("\nVERIFICATION RESULT\n")

    data_file.write("\nToTime: {}\n".format(veri_result.totime))
    data_file.write("\nNumber of steps: {}\n".format(veri_result.num_steps))
    data_file.write("\nStatus: {}\n".format(veri_result.status))
    data_file.write("\nUnsafe Point: {}\n".format(veri_result.unsafe_point))
    data_file.write("\nRuntime: {}\n".format(veri_result.runtime))
    data_file.close()

    return veri_result


def plot_unsafe_trace(veri_result):
    'plot unsafe trace'

    time_list = np.linspace(0.0, veri_result.totime, veri_result.num_steps + 1)
    m = veri_result.unsafe_trace[0].shape[0]
    n = len(veri_result.unsafe_trace)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    # get output trace
    for i in xrange(0, m):
        trace_i = np.zeros(n)
        unsafe_line_i = np.zeros(n)
        for j in xrange(0, n):
            trace_i_j = veri_result.unsafe_trace[j]
            trace_i[j] = trace_i_j[i]
            unsafe_line_i[j] = veri_result.unsafe_set.d[i]
        ax1.plot(time_list, trace_i)
        ax1.plot(time_list, unsafe_line_i, 'r')

    ax1.legend(['$x_{1}(t) + x_2(t) + x_3(t)$', 'US: Unsafe boundary'], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$ (seconds)', fontsize=20)
    plt.ylabel(r'$x_{1}(t) + x_2(t) + x_3(t), US$', fontsize=20)
    fig1.suptitle('Unsafe trace', fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    fig1.savefig('unsafe_trace.pdf')


def main():
    'main function'

    E, A, B, C = get_benchmark()
    dae_sys = construct_dae_automaton(E, A, B, C)
    dae_auto = convert_to_auto_dae(dae_sys)
    adm_projs = get_admissible_projectors(dae_auto)
    print "\nindex of the RL benchmark is: {}".format(len(adm_projs))
    decoupled_dae = decouple_auto_dae(dae_auto)
    basic_matrix = generate_consistent_basic_matrix(decoupled_dae)
    init_set = construct_init_set(basic_matrix)

    totime = 10.0
    num_steps = 1000
    solver_names = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']

    unsafe_set = construct_unsafe_set(dae_auto)
    veri_res = verify_safety(dae_auto, init_set, unsafe_set, totime, num_steps, solver_names[3])
    if veri_res.status == 'unsafe':
        plot_unsafe_trace(veri_res)    # plot unsafe trace if the system is unsafe

    # plot the reachable set
    reachset = veri_res.reach_set
    list_of_line_set_list = get_line_set(reachset, dae_auto.matrix_c.todense())
    plot_vline_set(list_of_line_set_list, totime, num_steps)


if __name__ == '__main__':

    main()
