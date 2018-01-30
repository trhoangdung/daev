'''
Two-interconnected roatating masses example run file
Dung Tran: Jan/2018
'''

from daev.daes import index_2_daes
from daev.engine.dae_automaton import DaeAutomation
from daev.engine.decoupling import DecouplingAutonomous
from daev.engine.set import LinearPredicate, ReachSet, RectangleSet2D, RectangleSet3D
from daev.engine.reachability import ReachSetAssembler
from daev.engine.verifier import Verifier
from daev.engine.printer import spaceex_printer
from daev.engine.plot import Plot
from daev.engine.projectors import admissible_projectors
from scipy.sparse import csc_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def get_benchmark():
    'get benchmark matrices'

    # TWO INTERCONNECTED ROTATING MASSES
    E, A, B, C = index_2_daes().two_interconnected_rotating_masses(1.0, 2.0)
    print "\n########################################################"
    print "\nTWO INTERCONNECTED ROTATING MASSES:"
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

    u_mat = np.array([[0, 1], [-1, 0]])    # user-defined input
    # u_mat = np.array([[-1, 0], [0, -2]])    # user-defined input
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

    # init_set_basic_matrix = basic_matrix[:, 2:4]
    init_set_basic_matrix = basic_matrix[:, 0:2]
    print "\ninit_set_basic_matrix shape = {}".format(init_set_basic_matrix.shape)
    # alpha_min = np.array([[0.1]])
    # alpha_max = np.array([[0.2]])
    alpha_min = np.array([[0.1], [1.0]])
    alpha_max = np.array([[0.2], [1.2]])

    print "\ninitial set basic matrix: \n{}".format(init_set_basic_matrix)
    print "\ninitial set alpha min: \n{}".format(alpha_min)
    print "\ninitial set alpha max: \n{}".format(alpha_max)

    init_set = ReachSet()
    init_set.set_basic_matrix(init_set_basic_matrix)
    init_set.set_alpha_min_max(alpha_min, alpha_max)

    return init_set


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

    for i in xrange(0, m):
        line_set_output_i = []
        for j in xrange(0, n):
            line_set_list = list_of_line_set_list[j]
            line_set_output_i.append(line_set_list[i])
            print "\noutput_{} at step {}: min = {}, max = {}".format(i, j, line_set_list[i].xmin, line_set_list[i].xmax)

        ax1 = pl1.plot_vlines(ax1, time_list.tolist(), line_set_output_i, colors=colors[i], linestyles='solid')

    ax1.legend([r'$z_{1}(t)$', r'$M_{2}(t)$', r'$u_{1}(t) = M_{1}$', r'$u_2(t) = M_{4}$'], fontsize=20)
    ax1.set_ylim(-2.0, 2.0)
    ax1.set_xlim(0, totime)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel(r'$z_{1}, M_{2}, u_{1}, u_{2}$', fontsize=20)
    fig1.suptitle('Individual Reachable set of $z_{1}$ and $M_{2}$', fontsize=25)
    fig1.savefig('individual_reachset_z1_M2_u1_u2.pdf')
    plt.show()


def plot_boxes(list_of_line_set_list):
    'plot reach set of two outputs as boxes'

    n = len(list_of_line_set_list)

    box_list = []
    for j in xrange(0, n):
        line_set_list = list_of_line_set_list[j]
        box_2d = RectangleSet2D()
        box_2d.set_bounds(line_set_list[0].xmin, line_set_list[0].xmax, line_set_list[1].xmin, line_set_list[1].xmax)
        box_list.append(box_2d)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    pl1 = Plot()
    ax1 = pl1.plot_boxes(ax1, box_list, facecolor='b', edgecolor='b')
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlim(-1.5, 1.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$M_2$', fontsize=20)
    plt.ylabel(r'$z_1$', fontsize=20)
    blue_patch = mpatches.Patch(color='b', label='$(z_{1}, M_{2})$')
    plt.legend(handles=[blue_patch])
    fig1.suptitle('Reachable set $(z_1, M_2)$', fontsize=25)
    plt.show()
    fig1.savefig('reachset_z1_M2.pdf')


def plot_boxes_vs_time(list_of_line_set_list, totime, num_steps):
    'plot boxes vs time'

    n = len(list_of_line_set_list)
    time_list = np.linspace(0.0, totime, num_steps + 1)

    box_list = []
    for j in xrange(0, n):
        line_set_list = list_of_line_set_list[j]
        box_3d = RectangleSet3D()
        box_3d.set_bounds(line_set_list[0].xmin, line_set_list[0].xmax, line_set_list[1].xmin, line_set_list[1].xmax, time_list[j], time_list[j])
        box_list.append(box_3d)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    pl2 = Plot()
    ax2 = pl2.plot_3d_boxes(ax2, box_list, facecolor='b', linewidth=0.5, edgecolor='b')
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_zlim(0, 10.5)
    ax2.tick_params(axis='z', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.set_xlabel('\n' + '$z_1$', fontsize=20, linespacing=2)
    ax2.set_ylabel('\n' + '$M_2$', fontsize=20, linespacing=3)
    ax2.set_zlabel('\n' + r'$t$ (second)', fontsize=25, linespacing=0.5)
    fig2.suptitle('Reachable Set $(z_1, M_2)$ vs. time $t$', fontsize=25)
    plt.tight_layout()
    fig2.savefig('reachset_vs_time.pdf')
    plt.show()


def construct_unsafe_set(dae_auto):
    'construct unsafe set'

    C = np.array([[0, 0, 1, 0, 0, 0]])    # M2 <= -0.7
    d = np.array([[-0.7]])
    print "\nunsafe_set 1:  matrix C = {}".format(C)
    print "\nunsafe_set 1:  vector d = {}".format(d)
    unsafe_set1 = LinearPredicate(C, d)

    C = np.array([[0, 0, 0, 1, 0, 0]])    # M3 <= -0.3
    d = np.array([[-1.0]])
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

    # get input traces
    input_mat = np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

    for i in xrange(0, 2):
        input_i_trace = np.zeros(n)
        for j in xrange(0, n):
            input_trace = np.dot(input_mat, veri_result.unsafe_state_trace[j])
            input_i_trace[j] = input_trace[i]

        ax1.plot(time_list, input_i_trace)

    ax1.legend(['Ouput $M_2(t)$', 'Unsafe boundary', 'Input $M_1(t)$', 'Input $M_4(t)$'], fontsize=20)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlim(0, 10.0)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$ (seconds)', fontsize=20)
    plt.ylabel(r'$M_1(t), M_2(t), M_4(t)$', fontsize=25)
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
    decoupled_dae = decouple_auto_dae(dae_auto)
    basic_matrix = generate_consistent_basic_matrix(decoupled_dae)
    init_set = construct_init_set(basic_matrix)

    totime = 10.0
    num_steps = 100
    solver_names = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']

    # print spaceex model
    spaceex_printer(decoupled_dae, init_set, totime, 0.01, 'RLC_circuit')

    unsafe_set = construct_unsafe_set(dae_auto)
    veri_res = verify_safety(dae_auto, init_set, unsafe_set, totime, num_steps, solver_names[3])

    reachset = veri_res[0].reach_set
    list_of_line_set_list = get_line_set(reachset, dae_auto.matrix_c.todense())
    plot_vline_set(list_of_line_set_list, totime, num_steps)
    plot_boxes(list_of_line_set_list)
    plot_boxes_vs_time(list_of_line_set_list, totime, num_steps)

    if veri_res[0].status == 'unsafe':
        plot_unsafe_trace(veri_res[0])


if __name__ == '__main__':

    main()
