# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import matplotlib.animation
import matplotlib.pyplot as plt
from dolfin import plot as original_plot
from rbnics.backends.common import TimeSeries
from rbnics.backends.online import OnlineFunction


def plot(obj, *args, **kwargs):
    if isinstance(obj, TimeSeries):
        is_time_series = True
    else:
        is_time_series = False
    if not is_time_series and isinstance(obj, OnlineFunction.Type()):
        is_truth_solution = False
    elif is_time_series and isinstance(obj[0], OnlineFunction.Type()):
        assert all(isinstance(obj_, OnlineFunction.Type()) for obj_ in obj)
        is_truth_solution = False
    else:
        is_truth_solution = True

    if is_time_series:
        if "every" in kwargs:
            obj = obj[::kwargs["every"]]
            del kwargs["every"]
        if "interval" in kwargs:
            anim_interval = kwargs["interval"]
            del kwargs["interval"]
        else:
            anim_interval = None

    if not is_truth_solution:
        assert "reduced_problem" in kwargs, (
            "Please use this method as plot(reduced_solution, reduced_problem=my_reduced_problem)"
            + " when plotting a reduced solution")
        if not is_time_series:
            N = obj.N
        else:
            N = obj[0].N
            assert all(obj_.N == N for obj_ in obj)
        reduced_problem = kwargs["reduced_problem"]
        del kwargs["reduced_problem"]
        basis_functions = reduced_problem.basis_functions[:N]
        truth_problem = reduced_problem.truth_problem
        if not is_time_series:
            obj = basis_functions * obj
        else:
            obj = [basis_functions * obj_ for obj_ in obj]
    elif "truth_problem" in kwargs:
        truth_problem = kwargs["truth_problem"]
        del kwargs["truth_problem"]
    else:
        truth_problem = None

    if "component" in kwargs:
        component = kwargs["component"]
        del kwargs["component"]
        if not is_time_series:
            obj = obj.sub(component)
        else:
            obj = [obj_.sub(component) for obj_ in obj]

    if truth_problem is not None and hasattr(truth_problem, "mesh_motion"):
        truth_problem.mesh_motion.move_mesh()

    if not is_time_series:
        output = original_plot(obj, *args, **kwargs)
    else:
        def animate(i):
            return original_plot(obj[i], *args, **kwargs).collections
        fig = plt.figure()
        output = matplotlib.animation.FuncAnimation(
            fig, animate, frames=len(obj), interval=anim_interval, repeat=False)
        try:
            from IPython.display import HTML
        except ImportError:
            pass
        else:
            output = HTML(output.to_html5_video())
            plt.close()

    if truth_problem is not None and hasattr(truth_problem, "mesh_motion"):
        truth_problem.mesh_motion.reset_reference()
    return output
