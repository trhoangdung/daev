'''
This module implements Plot class and its methods
Dung Tran: Nov/2017
'''

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from daev.engine.set import LineSet, RectangleSet2D, RectangleSet3D
from daev.engine.verifier import VerificationResult
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np


class Plot(object):
    'implements methods for ploting different kind of set'

    @staticmethod
    def plot_state_reachset_vs_time(verification_result):
        'plot individual state reachable set'

        assert isinstance(verification_result, VerificationResult)

        reachset = verification_result.reach_set
        k = len(reachset)
        n = verification_result.sys_dim
        m = verification_result.num_inputs
        totime = verification_result.totime
        num_steps = verification_result.num_steps

        In = np.eye(n)
        list_of_line_set_list = []
        for i in xrange(0, k):
            line_set = reachset[i].get_line_set(In)
            list_of_line_set_list.append(line_set)

        time_list = np.linspace(0.0, totime, num_steps + 1)

        for i in xrange(0, n - m):
            line_set_x_i = []
            for j in xrange(0, k):
                line_set_list = list_of_line_set_list[j]
                line_set_x_i.append(line_set_list[i])
                print "\nx_{} at step {}: min = {}, max = {}".format(i, j, line_set_list[i].xmin, line_set_list[i].xmax)

            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            pl = Plot()
            ax = pl.plot_vlines(ax, time_list.tolist(), line_set_x_i, colors='b', linestyles='solid')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$t$', fontsize=20)
            if m is not None and i >= n - m:
                plt.ylabel(r'$u_{}(t)$'.format(i - n + m), fontsize=20)
                fig.suptitle('Reachable set of $u_{}$'.format(i - n + m), fontsize=25)
            else:
                plt.ylabel(r'$x_{}(t)$'.format(i), fontsize=20)
                fig.suptitle('Reachable set of the state $x_{}$'.format(i), fontsize=25)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            if m is not None and i >= n - m:
                fig.savefig('u{}_vs_t.pdf'.format(i - n + m))
            else:
                fig.savefig('x{}_vs_t.pdf'.format(i))
            fig.show()
        plt.show()

    @staticmethod
    def plot_output_reachset_vs_time(verification_result, output_matrix):
        'plot individual output reach set defined by output matrix'

        assert isinstance(output_matrix, np.ndarray)
        assert isinstance(verification_result, VerificationResult)

        reachset = verification_result.reach_set
        k = len(reachset)
        totime = verification_result.totime
        num_steps = verification_result.num_steps

        m = output_matrix.shape[0]
        list_of_line_set_list = []
        for i in xrange(0, k):
            line_set = reachset[i].get_line_set(output_matrix)
            list_of_line_set_list.append(line_set)

        time_list = np.linspace(0.0, totime, num_steps + 1)

        for i in xrange(0, m):
            line_set_y_i = []
            for j in xrange(0, k):
                line_set_list = list_of_line_set_list[j]
                line_set_y_i.append(line_set_list[i])
                print "\ny_{} at step {}: min = {}, max = {}".format(i, j, line_set_list[i].xmin, line_set_list[i].xmax)

            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            pl = Plot()
            ax = pl.plot_vlines(ax, time_list.tolist(), line_set_y_i, colors='b', linestyles='solid')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('$t$', fontsize=20)
            plt.ylabel(r'$y_{}(t)$'.format(i), fontsize=20)
            fig.suptitle('Reachable set of the output $y_{}$'.format(i), fontsize=25)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            fig.savefig('y{}_vs_t.pdf'.format(i))
            fig.show()
        plt.show()

    @staticmethod
    def plot_unsafe_trace(verification_result):
        'plot unsafe trace if the system is unsafe'

        assert isinstance(verification_result, VerificationResult)

        totime = verification_result.totime
        num_steps = verification_result.num_steps
        time_list = np.linspace(0.0, totime, num_steps + 1)
        unsafe_set = verification_result.unsafe_set
        C = unsafe_set.C
        m = C.shape[0]    # number of output checked
        l = verification_result.num_inputs
        n = verification_result.sys_dim
        unsafe_state_trace = verification_result.unsafe_state_trace
        unsafe_trace = verification_result.unsafe_trace
        k = len(unsafe_trace)
        output_dynamics = Plot().get_dynamics(C)

        if verification_result.status == 'safe':
            print "\nThe system is safe, there is no unsafe trace"

        if verification_result.status == 'unsafe':

            for i in xrange(0, m):
                y_i = np.zeros(k)
                boundary_i = np.zeros(k)
                for j in xrange(0, k):
                    y_i_j = unsafe_trace[j]
                    y_i[j] = y_i_j[i]
                    boundary_i[j] = unsafe_set.d[i]

                fig = plt.figure(i)
                ax = fig.add_subplot(111)
                ax.plot(time_list, y_i)
                ax.plot(time_list, boundary_i, 'r')
                ax.legend(['$y_{}$'.format(i), 'unsafe boundary'])
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel('$t$', fontsize=20)
                plt.ylabel(r'$y_{}$'.format(i), fontsize=20)
                fig.suptitle('Unsafe trace: $y_{}$ = {} $\leq {}$'.format(i, output_dynamics[i], unsafe_set.d[i][0]), fontsize=25)
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                fig.savefig('unsafe_y_{}.pdf'.format(i))
                fig.show()

            for i in xrange(0, l):
                u_i = np.zeros(k)
                for j in xrange(0, k):
                    u_i_j = unsafe_state_trace[j]
                    u_i[j] = u_i_j[n - 1 - i]

                fig = plt.figure(i + m)
                ax = fig.add_subplot(111)
                ax.plot(time_list, u_i)
                ax.legend(['$u_{}(t)$'.format(i)])
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel('$t$', fontsize=20)
                plt.ylabel(r'$u_{}$'.format(i), fontsize=20)
                fig.suptitle('Unsafe input $u_{}(t)$'.format(i), fontsize=25)
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                fig.savefig('unsafe_input_u{}.pdf'.format(i))
                fig.show()

            plt.show()

    @staticmethod
    def get_dynamics(C):
        'print y = Cx'

        assert isinstance(C, np.ndarray)
        m, n = C.shape
        dynamics = []

        for i in xrange(0, m):
            yi = ''
            for j in xrange(0, n):
                if C[i, j] > 0:
                    cx = '${}x_{}$'.format(C[i, j], j)
                    if j == 0:
                        yi = '{} {}'.format(yi, cx)
                    else:
                        if yi != '':
                            yi = '{} + {}'.format(yi, cx)
                        else:
                            yi = '{}'.format(cx)
                elif C[i, j] < 0:
                    cx = '${}x_{}$'.format(-C[i, j], j)
                    yi = '{} - {}'.format(yi, cx)
            if yi == '':
                yi = '0'
            dynamics.append(yi)

        return dynamics

    @staticmethod
    def plot_boxes(ax, rectangle_set_list, facecolor, edgecolor):
        'plot reachable set using rectangle boxes'

        # return axes object to plot a figure
        n = len(rectangle_set_list)
        assert n > 0, 'empty set'
        assert isinstance(ax, Axes)

        for i in xrange(0, n):
            assert isinstance(rectangle_set_list[i], RectangleSet2D)

        xmin = []
        xmax = []
        ymin = []
        ymax = []

        for i in xrange(0, n):
            xmin.append(rectangle_set_list[i].xmin)
            xmax.append(rectangle_set_list[i].xmax)
            ymin.append(rectangle_set_list[i].ymin)
            ymax.append(rectangle_set_list[i].ymax)

            patch = Rectangle(
                (xmin[i],
                 ymin[i]),
                xmax[i] - xmin[i],
                ymax[i] - ymin[i],
                facecolor=facecolor,
                edgecolor=edgecolor,
                fill=True)
            ax.add_patch(patch)

        xmin.sort()
        xmax.sort()
        ymin.sort()
        ymax.sort()
        min_x = xmin[0]
        max_x = xmax[len(xmax) - 1]
        min_y = ymin[0]
        max_y = ymax[len(ymax) - 1]

        ax.set_xlim(min_x - 0.1 * abs(min_x), max_x + 0.1 * abs(max_x))
        ax.set_ylim(min_y - 0.1 * abs(min_y), max_y + 0.1 * abs(max_y))

        return ax

    @staticmethod
    def plot_vlines(ax, x_pos_list, lines_list, colors, linestyles):
        'plot vline at x'

        assert isinstance(ax, Axes)
        assert isinstance(x_pos_list, list)
        assert isinstance(lines_list, list)
        assert len(x_pos_list) == len(lines_list), 'inconsistent data, len_x = {} != len(line) = {}'.format(len(x_pos_list), len(lines_list))
        n = len(x_pos_list)
        ymin_list = []
        ymax_list = []
        for i in xrange(0, n):
            assert isinstance(lines_list[i], LineSet)
            ymin_list.append(lines_list[i].xmin)
            ymax_list.append(lines_list[i].xmax)

        ax.vlines(
            x_pos_list,
            ymin_list,
            ymax_list,
            colors=colors,
            linestyles=linestyles,
            linewidth=2)

        x_pos_list.sort()
        ymin_list.sort()
        ymax_list.sort()
        xmin = x_pos_list[0]
        xmax = x_pos_list[n - 1]
        ymin = ymin_list[0]
        ymax = ymax_list[n - 1]

        ax.set_xlim(xmin - 0.1 * abs(xmin), xmax + 0.1 * abs(xmax))
        ax.set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))

        return ax

    @staticmethod
    def plot_3d_boxes(ax, boxes_list, facecolor, linewidth, edgecolor):
        'plot 3d boxes contain reachable set'

        assert isinstance(boxes_list, list)
        assert isinstance(ax, Axes3D)
        for box in boxes_list:
            assert isinstance(box, RectangleSet3D)
            xmin = box.xmin
            xmax = box.xmax
            ymin = box.ymin
            ymax = box.ymax
            zmin = box.zmin
            zmax = box.zmax
            p1 = [xmin, ymin, zmin]
            p2 = [xmin, ymin, zmax]
            p3 = [xmin, ymax, zmin]
            p4 = [xmin, ymax, zmax]
            p5 = [xmax, ymin, zmin]
            p6 = [xmax, ymin, zmax]
            p7 = [xmax, ymax, zmin]
            p8 = [xmax, ymax, zmax]
            V = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
            #ax.scatter3D(V[:, 0], V[:, 1], V[:, 2])
            verts = [
                [
                    V[0], V[1], V[6], V[4]], [
                    V[0], V[2], V[6], V[4]], [
                    V[0], V[1], V[3], V[2]], [
                    V[4], V[5], V[7], V[6]], [
                        V[2], V[3], V[7], V[6]], [
                            V[1], V[3], V[7], V[5]]]

            ax.add_collection3d(
                Poly3DCollection(
                    verts,
                    facecolors=facecolor,
                    linewidths=linewidth,
                    edgecolors=edgecolor))

        x_min_list = []
        x_max_list = []
        y_min_list = []
        y_max_list = []
        z_min_list = []
        z_max_list = []

        for box in boxes_list:
            x_min_list.append(box.xmin)
            x_max_list.append(box.xmax)
            y_min_list.append(box.ymin)
            y_max_list.append(box.ymax)
            z_min_list.append(box.zmin)
            z_max_list.append(box.zmax)

        x_min_list.sort()
        x_max_list.sort()
        y_min_list.sort()
        y_max_list.sort()
        z_min_list.sort()
        z_max_list.sort()

        min_x = x_min_list[0]
        max_x = x_max_list[len(x_max_list) - 1]
        min_y = y_min_list[0]
        max_y = y_max_list[len(y_max_list) - 1]
        min_z = z_min_list[0]
        max_z = z_max_list[len(z_max_list) - 1]

        ax.set_xlim(min_x - 0.1 * abs(min_x), max_x + 0.1 * abs(max_x))
        ax.set_ylim(min_y - 0.1 * abs(min_y), max_y + 0.1 * abs(max_y))
        ax.set_ylim(min_z - 0.1 * abs(min_z), max_z + 0.1 * abs(max_z))

        return ax
