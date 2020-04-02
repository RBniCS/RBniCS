# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import plot as original_plot
from rbnics.backends.online import OnlineFunction

def plot(obj, *args, **kwargs):
    if isinstance(obj, OnlineFunction.Type()):
        assert "reduced_problem" in kwargs, (
            "Please use this method as plot(reduced_solution, reduced_problem=my_reduced_problem)"
            + " when plotting a reduced solution")
        N = obj.N
        reduced_problem = kwargs["reduced_problem"]
        del kwargs["reduced_problem"]
        basis_functions = reduced_problem.basis_functions[:N]
        truth_problem = reduced_problem.truth_problem
        obj = basis_functions * obj
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
