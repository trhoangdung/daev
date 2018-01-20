'''
Test reachability module
Dung Tran: 1/2018
'''

from daev.engine.reachability import ReachSetAssembler
from daev.engine.set import InitSet
from daev.engine.dae_automaton import DaeAutomation
from daev.engine.decoupling import DecouplingAutonomous
from daev.daes import index_1_daes, index_2_daes, index_3_daes
import numpy as np
from scipy.sparse import csc_matrix


def test_ode_sim():
    'test methods'

    A = np.array([[0.0, -1.0], [-1.0, 0.0]])
    init_vec = np.array([1, -0.25])
    final_time = 2.0
    num_steps = 10

    solution, runtime = ReachSetAssembler().ode_sim(
        A, init_vec, final_time, num_steps, 'dopri5')

    print "\nsolution = {}".format(solution)
    print "\nruntime = {}".format(runtime)


def test_reach_autonomous_ode():
    'test reach_autonomous_ode method'

    A = np.array([[0.0, -1.0], [-1.0, 0.0]])
    min_vec = [0.0, 0.1]
    max_vec = [0.0, 0.2]
    init_cond = InitSet()
    init_cond.set_params(min_vec, max_vec)
    initSet = init_cond.get_init_reachset()
    totime = 2.0
    num_steps = 20
    solver_name = 'dopri5'
    reach_set_list, runtime = ReachSetAssembler.reach_autonomous_ode(
        A, initSet, totime, num_steps, solver_name)
    print "\nreachable set computation runtime = {}".format(runtime)
    print "\nreachable set list = {}".format(reach_set_list)


def test_reach_autonomous_dae_index_1():
    'test reachable set computation of autonomous dae with index 1'

    # RCL circuit example
    E, A, B, C = index_1_daes().RLC_circuit(1.0, 1.0, 1.0)
    dae_sys = DaeAutomation()
    dae_sys.set_dynamics(
        csc_matrix(E),
        csc_matrix(A),
        csc_matrix(B),
        csc_matrix(C))
    u_mat = np.array([-1])

    dae_auto = dae_sys.convert_to_autonomous_dae(csc_matrix(u_mat))

    print "\ndae_auto matrix_e = {}".format(dae_auto.matrix_e.todense())
    print "\nrank of new E = {}".format(np.linalg.matrix_rank(dae_auto.matrix_e.todense()))
    print "\ndae_auto matrix_a = {}".format(dae_auto.matrix_a.todense())
    print "\ndae_auto matrix_c = {}".format(dae_auto.matrix_c.todense())

    decpl_dae, status = DecouplingAutonomous().get_decoupled_system(dae_auto)
    print "\ndecoupling status = {}".format(status)

    print "\ndecoupled dae_auto ode_a_mat = {}".format(decpl_dae.ode_matrix_a)
    print "\ndecoupled dae_auto alg_a_mat = {}".format(decpl_dae.alg_matrix_a)
    print "\nprojectors = {}".format(decpl_dae.projectors[0])

    # initial condition

    init_cond = InitSet()
    min_vec = [0.0, 0.1, 0.2, 0.1, 0.1]
    max_vec = [0.0, 0.2, 0.3, 0.1, 0.1]

    init_cond.set_params(min_vec, max_vec)
    initSet = init_cond.get_init_reachset()

    print "\ninit Set S(0) = {}".format(initSet.S)
    print "\ninit Set alpha_min = {}".format(initSet.alpha_min_vec)
    print "\ninit Set alpha_max = {}".format(initSet.alpha_max_vec)

    totime = 2.0
    num_steps = 10
    solver_name = 'dopri5'
    #reach_set_list, runtime = ReachSetAssembler().reach_autonomous_dae_index_1(
    #    decpl_dae, initSet, totime, num_steps, solver_name)

    # generate consistent initial condition

    cons_init_cond, runtime = ReachSetAssembler().generate_consistent_init_condition(decpl_dae)
    print "\nconsistent initial condition basic vectors: = {}".format(cons_init_cond)
    print "\nruntime for generate consistent initial basic vectors = {}".format(runtime)

def test_generate_consistent_init_condition_index_2():
    'test generating consistent initial condition method'

    # RL network, index-2
    E, A, B, C = index_2_daes().RL_network(1.0, 1.0)
    dae_sys = DaeAutomation()
    dae_sys.set_dynamics(
        csc_matrix(E),
        csc_matrix(A),
        csc_matrix(B),
        csc_matrix(C))
    u_mat = np.array([0])

    dae_auto = dae_sys.convert_to_autonomous_dae(csc_matrix(u_mat))

    print "\ndae_auto matrix_e = {}".format(dae_auto.matrix_e.todense())
    print "\nrank of new E = {}".format(np.linalg.matrix_rank(dae_auto.matrix_e.todense()))
    print "\ndae_auto matrix_a = {}".format(dae_auto.matrix_a.todense())
    print "\ndae_auto matrix_c = {}".format(dae_auto.matrix_c.todense())

    decpl_dae, status = DecouplingAutonomous().get_decoupled_system(dae_auto)
    print "\ndecoupling status = {}".format(status)

    print "\ndecoupled dae_auto ode_a_mat = {}".format(decpl_dae.ode_matrix_a)
    print "\nnorm of ode_a_mat = {}".format(np.linalg.norm(decpl_dae.ode_matrix_a))
    print "\ndecoupled dae_auto alg1_a_mat = {}".format(decpl_dae.alg1_matrix_a)
    print "\nnorm of alg1_a_mat = {}".format(np.linalg.norm(decpl_dae.alg1_matrix_a))
    print "\ndecoupled dae_auto alg2_a_mat = {}".format(decpl_dae.alg2_matrix_a)
    print "\nnorm of alg2_a_mat = {}".format(np.linalg.norm(decpl_dae.alg2_matrix_a))
    print "\ndecoupled dae_auto alg2_c_mat = {}".format(decpl_dae.alg2_matrix_c)
    print "\nprojectors = {}".format(decpl_dae.projectors)

    # generate consistent initial condition
    cons_init_cond, runtime = ReachSetAssembler().generate_consistent_init_condition(decpl_dae)
    print "\nconsistent initial condition basic vectors: = {}".format(cons_init_cond)
    print "\nruntime for generate consistent initial basic vectors = {}".format(runtime)

def test_generate_consistent_init_condition_index_3():
    'test generating consistent initial condition method'

    # RL network, index-2
    E, A, B, C = index_3_daes().car_pendulum(1.0, 1.0, 1.0)
    dae_sys = DaeAutomation()
    dae_sys.set_dynamics(
        csc_matrix(E),
        csc_matrix(A),
        csc_matrix(B),
        csc_matrix(C))
    u_mat = np.array([0])

    dae_auto = dae_sys.convert_to_autonomous_dae(csc_matrix(u_mat))

    print "\ndae_auto matrix_e = {}".format(dae_auto.matrix_e.todense())
    print "\nrank of new E = {}".format(np.linalg.matrix_rank(dae_auto.matrix_e.todense()))
    print "\ndae_auto matrix_a = {}".format(dae_auto.matrix_a.todense())
    print "\ndae_auto matrix_c = {}".format(dae_auto.matrix_c.todense())

    decpl_dae, status = DecouplingAutonomous().get_decoupled_system(dae_auto)
    print "\ndecoupling status = {}".format(status)

    print "\ndecoupled dae_auto ode_a_mat = {}".format(decpl_dae.ode_matrix_a)
    print "\nnorm of ode_a_mat = {}".format(np.linalg.norm(decpl_dae.ode_matrix_a))
    print "\ndecoupled dae_auto alg1_a_mat = {}".format(decpl_dae.alg1_matrix_a)
    print "\nnorm of alg1_a_mat = {}".format(np.linalg.norm(decpl_dae.alg1_matrix_a))
    print "\ndecoupled dae_auto alg2_a_mat = {}".format(decpl_dae.alg2_matrix_a)
    print "\nnorm of alg2_a_mat = {}".format(np.linalg.norm(decpl_dae.alg2_matrix_a))
    print "\ndecoupled dae_auto alg2_c_mat = {}".format(decpl_dae.alg2_matrix_c)
    print "\ndecoupled dae_auto alg3_a_mat = {}".format(decpl_dae.alg3_matrix_a)
    print "\nnorm of alg3_a_mat = {}".format(np.linalg.norm(decpl_dae.alg3_matrix_a))
    print "\ndecoupled dae_auto alg3_c_mat = {}".format(decpl_dae.alg3_matrix_c)
    print "\ndecoupled dae_auto alg3_d_mat = {}".format(decpl_dae.alg3_matrix_d)
    print "\nprojectors = {}".format(decpl_dae.projectors)

    # generate consistent initial condition
    cons_init_cond, runtime = ReachSetAssembler().generate_consistent_init_condition(decpl_dae)
    print "\nconsistent initial condition basic vectors: = {}".format(cons_init_cond)
    print "\nruntime for generate consistent initial basic vectors = {}".format(runtime)


if __name__ == '__main__':
    # test_ode_sim()
    test_reach_autonomous_ode()
    # test_reach_autonomous_dae_index_1()
    # test_generate_consistent_init_condition_index_2()
    # test_generate_consistent_init_condition_index_3()
