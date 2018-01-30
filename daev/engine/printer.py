'''
This module implements some methods to print a decoupled dae system into Spaceex format
Dung Tran Jan/2018
'''

import numpy as np
from daev.engine.decoupling import AutonomousDecoupledIndexOne, AutonomousDecoupledIndexTwo
from daev.engine.decoupling import AutonomousDecoupledIndexThree
from daev.engine.set import ReachSet


def get_dynamics(C):
    'print y = Cx'

    assert isinstance(C, np.ndarray)
    m, n = C.shape
    dynamics = []

    for i in xrange(0, m):
        yi = ''
        for j in xrange(0, n):
            if C[i, j] > 0:
                cx = '{}*x{}'.format(C[i, j], j)
                if j == 0:
                    yi = '{} {}'.format(yi, cx)
                else:
                    if yi != '':
                        yi = '{} + {}'.format(yi, cx)
                    else:
                        yi = '{}'.format(cx)
            elif C[i, j] < 0:
                cx = '{}*x{}'.format(-C[i, j], j)
                yi = '{} - {}'.format(yi, cx)
        if yi == '':
            yi = '0'
        dynamics.append(yi)

    return dynamics


def print_spaceex_xml_file_autonomous_ode(A, C, file_name):
    'print spaceex xml file for dot{x} = Ax, y = Cx'

    assert isinstance(
        A, np.ndarray) and A.shape[0] == A.shape[1], 'error: A is not an ndarray or A is not a square matrix'
    assert isinstance(C, np.ndarray), 'error: C is not an ndarray'
    assert C.shape[1] == A.shape[0], 'error: inconsistent between A and C'
    assert isinstance(file_name, str), 'error: file name should be a string'

    n = A.shape[0]
    m = C.shape[0]
    xml_file = open(file_name, 'w')

    xml_file.write('<?xml version="1.0" encoding="iso-8859-1"?>\n')
    xml_file.write(
        '<sspaceex xmlns="http://www-verimag.imag.fr/xml-namespaces/sspaceex" version="0.2" math="SpaceEx">\n')

    # print core component
    xml_file.write('  <component id="core_component">\n')

    for i in xrange(0, n):
        xml_file.write(
            '    <param name="x{}" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n'.format(i))
    for i in xrange(0, m):
        xml_file.write(
            '    <param name="y{}" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n'.format(i))
    xml_file.write(
        '    <param name="t" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n')
    xml_file.write(
        '    <param name="stoptime" type="real" local="false" d1="1" d2="1" dynamics="const"/>\n')

    xml_file.write(
        '    <location id="1" name="Model" x="362.0" y="430.0" width="426.0" height="610.0">\n')

    # print invariant
    xml_file.write('      <invariant>\n')
    xml_file.write('        t &lt;= stoptime\n')
    output_dynamics = get_dynamics(C)
    for i in xrange(0, m):
        xml_file.write(
            '        &amp;y{} == {}\n'.format(
                i, output_dynamics[i]))
    xml_file.write('      </invariant>\n')

    # print flow
    xml_file.write('      <flow>\n')
    xml_file.write('        t\' == 1\n')
    for i in xrange(0, n):
        C1 = A[i]
        xi_dynamics = get_dynamics(C1)
        xml_file.write('        &amp;x{}\' == {}\n'.format(i, xi_dynamics[0]))

    xml_file.write('      </flow>\n')
    xml_file.write('    </location>\n')
    xml_file.write('  </component>\n')

    # print system
    xml_file.write('  <component id="sys">\n')
    for i in xrange(0, n):
        xml_file.write(
            '    <param name="x{}" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>\n'.format(i))
    for i in xrange(0, m):
        xml_file.write(
            '    <param name="y{}" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>\n'.format(i))
    xml_file.write(
        '    <param name="t" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>\n')
    xml_file.write(
        '    <param name="stoptime" type="real" local="false" d1="1" d2="1" dynamics="const" controlled="true"/>\n')

    xml_file.write('    <bind component="core_component" as="model">\n')
    for i in xrange(0, n):
        xml_file.write('      <map key="x{}">x{}</map>\n'.format(i, i))
    for i in xrange(0, m):
        xml_file.write('      <map key="y{}">y{}</map>\n'.format(i, i))
    xml_file.write('      <map key="t">t</map>\n')
    xml_file.write('      <map key="stoptime">stoptime</map>\n')
    xml_file.write('    </bind>\n')

    xml_file.write('  </component>')

    xml_file.write('</sspaceex>')

    xml_file.close()

    return xml_file


def print_spaceex_cfg_file_autonomous_ode(
        xmin_vec, xmax_vec, ymin_vec, ymax_vec, stoptime, step, file_name):
    'print configuration file for spaceex model of autonomous ODE: dot{x} = Ax, y = Cx'

    assert isinstance(xmin_vec, list), 'error: xmin_vec is not a list'
    assert isinstance(xmax_vec, list), 'error: xmax_vec is not a list'
    assert len(xmin_vec) == len(
        xmax_vec), 'error: inconsistency between xmin_vec and xmax_vec'

    assert isinstance(ymin_vec, list), 'error: ymin_vec is not a list'
    assert isinstance(ymax_vec, list), 'error: ymax_vec is not a list'
    assert len(ymin_vec) == len(
        ymax_vec), 'error: inconsistency between ymin_vec and ymax_vec'
    assert isinstance(file_name, str)

    cfg_file = open(file_name, 'w')

    cfg_file.write('# analysis option \n')
    cfg_file.write('system = "sys"\n')

    # init string
    init_str = ''
    n = len(xmin_vec)
    m = len(ymin_vec)
    for i in xrange(0, n):
        init_str = '{} x{} >= {} & x{} <= {} &'.format(
            init_str, i, xmin_vec[i], i, xmax_vec[i])
    for i in xrange(0, m):
        init_str = '{} y{} >= {} & y{} <= {} &'.format(
            init_str, i, ymin_vec[i], i, ymax_vec[i])
    init_str = '{} t == 0 & stoptime == {}'.format(init_str, stoptime)

    cfg_file.write('initially = "{}"\n'.format(init_str))
    cfg_file.write('scenario = "supp" \n')
    cfg_file.write('directions = "box"\n')
    cfg_file.write('sampling-time = {}\n'.format(step))
    cfg_file.write('time-horizon = {}\n'.format(stoptime))
    cfg_file.write('iter-max = 10\n')
    output = ''
    for i in xrange(0, m):
        if i < m - 1:
            output = '{}y{}, '.format(output, i)
        else:
            output = '{}y{}'.format(output, i)
    cfg_file.write('output-variables = "t, {}" \n'.format(output))
    cfg_file.write('output-format = "GEN"\n')
    cfg_file.write('rel-err = 1.0e-8\n')
    cfg_file.write('abs-err = 1.0e-12\n')

    cfg_file.close()

    return cfg_file


def spaceex_printer(decoupled_sys, init_set, stoptime, step, file_name):
    'print spaceex model from decoupled autonomous dae and init_set'

    assert isinstance(
        decoupled_sys,
        AutonomousDecoupledIndexOne) or isinstance(
        decoupled_sys,
        AutonomousDecoupledIndexTwo) or isinstance(
            decoupled_sys,
        AutonomousDecoupledIndexThree)

    assert isinstance(init_set, ReachSet)
    assert isinstance(file_name, str)
    xml_file_name = '{}.xml'.format(file_name)
    cfg_file_name = '{}.cfg'.format(file_name)

    C = decoupled_sys.reach_set_projector
    print "\nreachset projector = {}".format(C)

    A = decoupled_sys.N1
    xml_file = print_spaceex_xml_file_autonomous_ode(A, C, xml_file_name)    # print xml file

    In = np.eye(A.shape[0])
    state_line_set = init_set.get_line_set(In)
    output_line_set = init_set.get_line_set(C)

    n = len(state_line_set)
    xmin = []
    xmax = []
    for i in xrange(0, n):
        xmin.append(state_line_set[i].xmin)
        xmax.append(state_line_set[i].xmax)

    m = len(output_line_set)
    ymin = []
    ymax = []
    for i in xrange(0, m):
        ymin.append(output_line_set[i].xmin)
        ymax.append(output_line_set[i].xmax)

    cfg_file = print_spaceex_cfg_file_autonomous_ode(xmin, xmax, ymin, ymax, stoptime, step, cfg_file_name)

    return xml_file, cfg_file
