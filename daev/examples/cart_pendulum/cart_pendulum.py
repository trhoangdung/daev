'''
Cart-pendulum example run file
Dung Tran: Jan/2018
'''

from daev.daes import index_3_daes
from daev.engine.dae_automaton import DaeAutomation
from daev.engine.decoupling import DecouplingAutonomous
from scipy.sparse import csc_matrix
import numpy as np


def get_cart_pendulum():
    'get cart-pendulum matrices'

    # Cart Pendulum benchmark
    E, A, B, C = index_3_daes().car_pendulum(2, 1, 1)
    print "\n########################################################"
    print "\nCAR PENDULUM:"
    print "\ndimensions: {}".format(E.shape[0])
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E.todense(), A.todense(), B.todense(), C.todense())

    return E, A, B, C

def construct_dae_automaton():
    'create dae automaton for car pedulum benchmark'
    E, A, B, C = get_cart_pendulum()
    dae_sys = DaeAutomation()
    dae_sys.set_dynamics(E, A, B, C)
    return dae_sys

def convert_to_auto_dae():
    'convert cart pendulum dae system to autonomous dae automaton'

    u_mat = np.array([-1])
    dae_sys = construct_dae_automaton()
    dae_auto = dae_sys.convert_to_autonomous_dae(csc_matrix(u_mat))

    print "\ndae_auto matrix_e = {}".format(dae_auto.matrix_e.todense())
    print "\nrank of new E = {}".format(np.linalg.matrix_rank(dae_auto.matrix_e.todense()))
    print "\ndae_auto matrix_a = {}".format(dae_auto.matrix_a.todense())
    print "\ndae_auto matrix_c = {}".format(dae_auto.matrix_c.todense())

    return dae_auto

def decouple_auto_dae():
    'decoupling autonomous car-pendulum dae system'

    dae_auto = convert_to_auto_dae()
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

def generate_consistent_basic_matrix():
    'generate an consistent basic_matrix for initial condition'

    decoupled_dae = decouple_auto_dae()
    basic_matrix = decoupled_dae.generate_consistent_init_set_basic_matrix()
    print "\nconsistent initial set basic matrix: \n{}".format(basic_matrix)
    print "\nnorm of basic matrix = {}".format(np.linalg.norm(basic_matrix))
    print "\nbasic matrix shape = {}".format(basic_matrix.shape)

    return basic_matrix

if __name__ == '__main__':
    # get_cart_pendulum()
    # construct_dae_automaton()
    # convert_to_auto_dae()
    # decouple_auto_dae()
    generate_consistent_basic_matrix()
