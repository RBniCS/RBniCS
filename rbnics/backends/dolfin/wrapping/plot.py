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

from dolfin import plot as original_plot
from rbnics.backends.online import OnlineFunction

def plot(obj, *args, **kwargs):
    if isinstance(obj, OnlineFunction.Type()):
        assert "reduced_problem" in kwargs, "Please use this method as plot(reduced_solution, reduced_problem=my_reduced_problem) when plotting a reduced solution"
        N = obj.N
        reduced_problem = kwargs["reduced_problem"]
        del kwargs["reduced_problem"]
        basis_functions = reduced_problem.basis_functions[:N]
        truth_problem = reduced_problem.truth_problem
        obj = basis_functions*obj
    elif "truth_problem" in kwargs:
        truth_problem = kwargs["truth_problem"]
        del kwargs["truth_problem"]
    else:
        truth_problem = None
    if truth_problem is not None and hasattr(truth_problem, "mesh_motion"):
        truth_problem.mesh_motion.move_mesh()
    original_plot(obj, *args, **kwargs)
    if truth_problem is not None and hasattr(truth_problem, "mesh_motion"):
        truth_problem.mesh_motion.reset_reference()
