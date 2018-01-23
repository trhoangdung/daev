'''
Cart-pendulum example run file
Dung Tran: Jan/2018
'''

from daev.daes import index_3_daes
from daev.engine.dae_automaton import DaeAutomation
from daev.engine.decoupling import DecouplingAutonomous
from daev.engine.set import ReachSet
from daev.engine.reachability import ReachSetAssembler
from daev.engine.plot import Plot
from scipy.sparse import csc_matrix
import numpy as np
import matplotlib.pyplot as plt


def get_cart_pendulum():
    'get cart-pendulum matrices'

    # Cart Pendulum benchmark
    E, A, B, C = index_3_daes().car_pendulum(2, 1, 1)
    print "\n########################################################"
    print "\nCAR PENDULUM:"
    print "\ndimensions: {}".format(E.shape[0])
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E.todense(), A.todense(), B.todense(), C.todense())

    return E, A, B, C


def construct_dae_automaton(E, A, B, C):
    'create dae automaton for car pedulum benchmark'

    dae_sys = DaeAutomation()
    dae_sys.set_dynamics(E, A, B, C)
    return dae_sys


def convert_to_auto_dae(dae_sys):
    'convert cart pendulum dae system to autonomous dae automaton'

    u_mat = np.array([-2])    # user-defined input
    dae_auto = dae_sys.convert_to_autonomous_dae(csc_matrix(u_mat))
    print "\ndae_auto matrix_e = {}".format(dae_auto.matrix_e.todense())
    print "\nrank of new E = {}".format(np.linalg.matrix_rank(dae_auto.matrix_e.todense()))
    print "\ndae_auto matrix_a = {}".format(dae_auto.matrix_a.todense())
    print "\ndae_auto matrix_c = {}".format(dae_auto.matrix_c.todense())

    return dae_auto


def decouple_auto_dae(dae_auto):
    'decoupling autonomous car-pendulum dae system'

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

    # init_set_basic_matrix = basic_matrix[:, 2:4]
    init_set_basic_matrix = basic_matrix[:, 0:2]
    print "\ninit_set_basic_matrix shape = {}".format(init_set_basic_matrix.shape)
    # alpha_min = np.array([[0.1]])
    # alpha_max = np.array([[0.2]])
    alpha_min = np.array([[0.1], [0.8]])
    alpha_max = np.array([[0.2], [1.2]])

    print "\ninitial set basic matrix: \n{}".format(init_set_basic_matrix)
    print "\ninitial set alpha min: \n{}".format(alpha_min)
    print "\ninitial set alpha max: \n{}".format(alpha_max)

    init_set = ReachSet()
    init_set.set_basic_matrix(init_set_basic_matrix)
    init_set.set_alpha_min_max(alpha_min, alpha_max)

    return init_set


def compute_reachable_set(dae_auto, init_set, totime, num_steps, solver_name):
    'compute reachable set'

    reachset, decoupling_time, reachset_computation_time = ReachSetAssembler.reach_autonomous_dae(dae_auto, init_set, totime, num_steps, solver_name)
    print "\nlength of reachset = {}".format(len(reachset))
    print "\ndecoupling time = {}".format(decoupling_time)
    print "\nreachable set computation time = {}".format(reachset_computation_time)

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
    'plot reach set of each output'

    n = len(list_of_line_set_list)    # number of line_set list
    m = len(list_of_line_set_list[0])    # number of outputs
    time_list = np.linspace(0.0, totime, num_steps + 1)
    print "\ntype of time_list = {}".format(type(time_list))
    print "\ntime_list = {}".format(time_list)

    for i in xrange(0, m):
        line_set_output_i = []
        for j in xrange(0, n):
            line_set_list = list_of_line_set_list[j]
            line_set_output_i.append(line_set_list[i])
            print "\noutput_{} at step {}: min = {}, max = {}".format(i, j, line_set_list[i].xmin, line_set_list[i].xmax)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        pl1 = Plot()
        ax1 = pl1.plot_vlines(ax1, time_list.tolist(), line_set_output_i, colors='b', linestyles='solid')
        ax1.legend([r'$y_{}(t)$'.format(i)])
        ax1.set_ylim(-0.2, 2.0)
        ax1.set_xlim(0, totime)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('$t$', fontsize=20)
        plt.ylabel(r'$y_{}$'.format(i), fontsize=20)
        fig1.suptitle('Simulation-equivalent reachable set of $y_{}$'.format(i), fontsize=25)
        fig1.savefig('dreachset_y_{}.pdf'.format(i))
        plt.show()


def main():
    'main function'

    E, A, B, C = get_cart_pendulum()
    dae_sys = construct_dae_automaton(E, A, B, C)
    dae_auto = convert_to_auto_dae(dae_sys)
    decoupled_dae = decouple_auto_dae(dae_auto)
    basic_matrix = generate_consistent_basic_matrix(decoupled_dae)
    init_set = construct_init_set(basic_matrix)

    totime = 10.0
    num_steps = 100
    solver_names = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']    # similar to ode45 mathlab

    reachset = compute_reachable_set(dae_auto, init_set, totime, num_steps, solver_names[3])

    list_of_line_set_list = get_line_set(reachset, dae_auto.matrix_c.todense())

    plot_vline_set(list_of_line_set_list, totime, num_steps)

if __name__ == '__main__':

    main()
