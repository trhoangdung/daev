'''
Stokes-equation example run file
Dung Tran: Jan/2018
'''

from daev.daes import index_2_daes
from daev.engine.dae_automaton import DaeAutomation
from daev.engine.decoupling import DecouplingAutonomous
from daev.engine.set import LinearPredicate, ReachSet, RectangleSet2D, RectangleSet3D
from daev.engine.reachability import ReachSetAssembler
from daev.engine.verifier import Verifier
from daev.engine.plot import Plot
from scipy.sparse import csc_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def construct_unsafe_set(dae_auto):
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
    'plot reach set of individual outputs'

    n = len(list_of_line_set_list)    # number of line_set list
    m = len(list_of_line_set_list[0])    # number of outputs
    time_list = np.linspace(0.0, totime, num_steps + 1)
    print "\ntime_list = {}".format(time_list)

    colors = ['b', 'g']
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

    ax1.legend([r'$v_{x}^c(t)$', r'$v_{y}^c(t)$'])
    ax1.set_ylim(-0.7, 0.05)
    ax1.set_xlim(0, totime)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel(r'$v_{x}^c, v_{y}^c$', fontsize=20)
    fig1.suptitle('Individual Reachable set of $v_{x}^c$ and $v_{y}^c$', fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig1.savefig('individual_reachset_vcx_vcy.pdf')
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
    ax1.set_ylim(-0.1, 0.05)
    ax1.set_xlim(-0.7, 0.01)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$v_{x}^c$', fontsize=20)
    plt.ylabel(r'$v_{y}^c$', fontsize=20)
    blue_patch = mpatches.Patch(color='b', label='$(v_{x}^c, v_{y}^c)$')
    plt.legend(handles=[blue_patch])
    fig1.suptitle('Reachable set $(v_{x}^c, v_{y}^c)$', fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    fig1.savefig('reachset_vcx_vcy.pdf')


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
    ax2.set_xlim(-0.7, 0.01)
    ax2.set_ylim(-0.15, 0.05)
    ax2.set_zlim(0, 0.4)
    ax2.tick_params(axis='z', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.set_xlabel('\n' + '$v_{x}^c$', fontsize=20, linespacing=2)
    ax2.set_ylabel('\n' + '$v_{y}^c$', fontsize=20, linespacing=3)
    ax2.set_zlabel('\n' + r'$t$ (second)', fontsize=20, linespacing=1)
    fig2.suptitle('Reachable Set $(v_{x}^c, v_{y}^c)$ vs. time $t$', fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig2.savefig('reachset_vcx_vcy_vs_time.pdf')
    plt.show()


def verify_safety(dae_auto, init_set, unsafe_set, totime, num_steps, solver_name):
    'verify the safety of the system'

    veri_result = Verifier().check_safety(dae_auto, init_set, unsafe_set, totime, num_steps, solver_name)
    print "\nsafety status = {}".format(veri_result.status)
    print "\nruntime = {}".format(veri_result.runtime)
    if veri_result.status == 'unsafe':
        print "\nunsafe_point: output = {}, t = {} seconds, fes_alpha = {}".format(veri_result.unsafe_point[0], veri_result.unsafe_point[1], veri_result.unsafe_point[2])

    return veri_result


def plot_unsafe_trace(veri_result, dae_auto_dimension):
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
    input_mat = np.zeros(dae_auto_dimension)
    input_mat[dae_auto_dimension - 1] = 1

    input_trace = np.zeros(n)
    for j in xrange(0, n):
        input_j = np.dot(input_mat, np.transpose(veri_result.unsafe_state_trace[j]))
        input_trace[j] = input_j

    ax1.plot(time_list, input_trace)

    ax1.legend(['$-v_{x}^c - v_{y}^c$', 'Unsafe boundary', 'Input $u(t)$'])
    ax1.set_ylim(-0.1, 0.7)
    ax1.set_xlim(0, 0.4)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$ (seconds)', fontsize=20)
    plt.ylabel(r'$-v_{x}^c - v_{y}^c$', fontsize=20)
    fig1.suptitle('Unsafe trace', fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    fig1.savefig('unsafe_trace.pdf')


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
        unsafe_set = construct_unsafe_set(dae_auto)
        veri_res = verify_safety(dae_auto, init_set, unsafe_set, totime, num_steps, solver_names[3])
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

    print "\nvode times = {}".format(vode_times)

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
    plt.xlabel('$N$ (System dimension)', fontsize=20)
    plt.ylabel(r'Computation time', fontsize=20)
    fig1.suptitle('Reachable set computation time vs. solvers', fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    fig1.savefig('reach_time_vs_solvers.pdf')


def get_reach_time_vs_num_steps():
    'reachable set computation time vs number of steps'

    E, A, B, C = get_benchmark(1.0, 18)
    dae_sys = construct_dae_automaton(E, A, B, C)
    dae_auto = convert_to_auto_dae(dae_sys)
    decoupled_dae = decouple_auto_dae(dae_auto)
    basic_matrix = generate_consistent_basic_matrix(decoupled_dae)
    init_set = construct_init_set(basic_matrix)

    totime = 0.4
    num_steps = [100, 200, 400, 600, 800, 1000]
    solver = 'dopri5'
    reach_times = []
    for num in num_steps:
        _, _, reachset_computation_time = ReachSetAssembler.reach_autonomous_dae(dae_auto, init_set, totime, num, solver)
        reach_times.append(('solver={}, dimension={}, num_steps={}, totime={}'.format(solver, dae_auto.matrix_a.shape[0], num, totime), reachset_computation_time))
        print "\nsolver={}, dimension={}, num_steps={}, totime={} -> reach_time = {}".format(solver, dae_auto.matrix_a.shape[0], num, totime, reachset_computation_time)

    data_file = open('reach_time_vs_num_steps.dat', 'w')
    data_file.write("\n    REACHABLE SET COMPUTATION TIME VERSUS NUMBER OF TIME STEPS\n")

    for i in xrange(len(reach_times)):
        data_file.write("\n{}\n".format(reach_times[i]))
    data_file.close()

    return reach_times


def get_reach_time_vs_final_time():
    'reachable set computation time vs number of steps'

    E, A, B, C = get_benchmark(1.0, 18)
    dae_sys = construct_dae_automaton(E, A, B, C)
    dae_auto = convert_to_auto_dae(dae_sys)
    decoupled_dae = decouple_auto_dae(dae_auto)
    basic_matrix = generate_consistent_basic_matrix(decoupled_dae)
    init_set = construct_init_set(basic_matrix)

    totimes = [0.4, 0.6, 0.8, 1.0]
    num_steps = 1000
    solver = 'dopri5'
    reach_times = []
    for final_time in totimes:
        _, _, reachset_computation_time = ReachSetAssembler.reach_autonomous_dae(dae_auto, init_set, final_time, num_steps, solver)
        reach_times.append(('solver={}, dimension={}, num_steps={}, totime={}'.format(solver, dae_auto.matrix_a.shape[0], num_steps, final_time), reachset_computation_time))
        print "\nsolver={}, dimension={}, num_steps={}, totime={} -> reach_time = {}".format(solver, dae_auto.matrix_a.shape[0], num_steps, final_time, reachset_computation_time)

    data_file = open('reach_time_vs_final_time.dat', 'w')
    data_file.write("\n    REACHABLE SET COMPUTATION TIME VERSUS FINAL TIME\n")

    for i in xrange(len(reach_times)):
        data_file.write("\n{}\n".format(reach_times[i]))
    data_file.close()

    return reach_times


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


def main():
    'main function'

    # small-dimensional stokes-equation case, this takes about 5 minutes
    print "\nAnalyze small-dimensional stokes-equation, take about 5 minutes to complete"
    E, A, B, C = get_benchmark(1.0, 2)    # small case, dim = 17
    dae_sys = construct_dae_automaton(E, A, B, C)
    dae_auto = convert_to_auto_dae(dae_sys)
    decoupled_dae = decouple_auto_dae(dae_auto)
    basic_matrix = generate_consistent_basic_matrix(decoupled_dae)
    init_set = construct_init_set(basic_matrix)

    totime = 0.4
    num_steps = 50
    solver_names = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']    # similar to ode45 mathlab

    # reachset = compute_reachable_set(dae_auto, init_set, totime, num_steps, solver_names[3])
    # list_of_line_set_list = get_line_set(reachset, dae_auto.matrix_c.todense())
    # plot_vline_set(list_of_line_set_list, totime, num_steps)
    # plot_boxes(list_of_line_set_list)
    # plot_boxes_vs_time(list_of_line_set_list, totime, num_steps)

    # unsafe_set = construct_unsafe_set(dae_auto)
    # veri_res = verify_safety(dae_auto, init_set, unsafe_set, totime, num_steps, solver_names[3])
    # if veri_res.status == 'unsafe':
        # plot_unsafe_trace(veri_res, dae_auto.matrix_a.shape[0])

    print "\n##############TIME COMPLEXITY ANALYSIS#################"
    print "\ntaking about 1.5 hour to complete...."
    print "\nplease select one in the following options to go, you can select all"

    # verification_time = get_verification_time()    # this takes about 45 minutes
    # solver_times = get_ode_solvers_time()    # this takes about 45 minutes
    # plot_reach_times_vs_solvers(solver_times)

    # reach_time_vs_num_steps = get_reach_time_vs_num_steps()    # this takes about 5 minutes
    # reach_time_vs_final_time = get_reach_time_vs_final_time()    # this takes about 5 minutes
    # get_reach_time_vs_num_basic_vectors()    # this takes about 3 mintues


if __name__ == '__main__':

    main()
