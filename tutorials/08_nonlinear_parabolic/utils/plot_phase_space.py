# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import matplotlib.pyplot as plt


def plot_phase_space(solution_over_time, reduced_solution_over_time, basis_functions, x):
    fig = plt.figure()
    (all_u1, all_u2) = (list(), list())
    (all_reduced_u1, all_reduced_u2) = (list(), list())
    assert len(solution_over_time) == len(reduced_solution_over_time)
    for (solution, reduced_solution) in zip(solution_over_time, reduced_solution_over_time):
        (u1, u2) = solution.split()
        all_u1.append(u1(x))
        all_u2.append(u2(x))
        (reduced_u1, reduced_u2) = (basis_functions * reduced_solution).split()
        all_reduced_u1.append(reduced_u1(x))
        all_reduced_u2.append(reduced_u2(x))
    offline_line, = plt.plot(all_u1, all_u2, label="Truth solution")
    online_line, = plt.plot(all_reduced_u1, all_reduced_u2, label="Reduced solution")
    plt.legend(handles=[offline_line, online_line])
    plt.close()
    return fig
