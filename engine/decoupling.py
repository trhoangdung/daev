'''
This module implements decomposition techniques for high-index DAE
Dung Tran: Dec/2017
'''

from engine.dae_automaton import DaeAutomation


class Decomposition(object):
    'Decomposition techniques for high-index DAE'

    @staticmethod
    def kron_reduction(dae_automaton):
        'implement kron reduction for index-1 dae'

        pass

    @staticmethod
    def Marz_decoupling_index_1(dae_automaton):
        'implement Marz decoupling for index 1 dae automaton'

        # computation approach is based on the paper:
        # "An efficient projector-based passivity test for descritpor", Z.Zhang and N.Wong, 2010

        assert isinstance(dae_automaton, DaeAutomation)

        matrix_e = dae_automaton.matrix_e
        matrix_a = dae_automaton.matrix_a
        matrix_b = dae_automaton.matrix_b
        matrix_c = dae_automaton.matrix_c
