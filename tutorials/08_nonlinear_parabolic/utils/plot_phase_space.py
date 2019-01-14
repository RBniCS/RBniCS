# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

import os
import matplotlib.pyplot as plt

def plot_phase_space(solution_over_time, reduced_solution_over_time, basis_functions, x, folder, filename):
    plt.figure()
    (all_u1, all_u2) = (list(), list())
    (all_reduced_u1, all_reduced_u2) = (list(), list())
    assert len(solution_over_time) == len(reduced_solution_over_time)
    for (solution, reduced_solution) in zip(solution_over_time, reduced_solution_over_time):
        (u1, u2) = solution.split()
        all_u1.append(u1(x))
        all_u2.append(u2(x))
        (reduced_u1, reduced_u2) = (basis_functions*reduced_solution).split()
        all_reduced_u1.append(reduced_u1(x))
        all_reduced_u2.append(reduced_u2(x))
    offline_line, = plt.plot(all_u1, all_u2, label="Offline solution")
    online_line, = plt.plot(all_reduced_u1, all_reduced_u2, label="Online solution")
    plt.legend(handles=[offline_line, online_line])
    plt.savefig(os.path.join(folder, filename + ".png"))
