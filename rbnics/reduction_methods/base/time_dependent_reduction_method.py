# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import inspect
from rbnics.backends import assign, TimeSeries
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators
from rbnics.utils.test import PatchInstanceMethod


@RequiredBaseDecorators(None)
def TimeDependentReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class TimeDependentReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        def postprocess_snapshot(self, snapshot_over_time, snapshot_index):
            postprocessed_snapshot = TimeSeries(snapshot_over_time)
            for (k, t) in enumerate(snapshot_over_time.stored_times()):
                self.reduced_problem.set_time(t)
                postprocessed_snapshot_k = DifferentialProblemReductionMethod_DerivedClass.postprocess_snapshot(
                    self, snapshot_over_time[k], snapshot_index)
                postprocessed_snapshot.append(postprocessed_snapshot_k)
            return postprocessed_snapshot

        def _patch_truth_solve(self, force, **kwargs):
            if "with_respect_to" in kwargs:
                assert inspect.isfunction(kwargs["with_respect_to"])
                other_truth_problem = kwargs["with_respect_to"](self.truth_problem)

                def patched_truth_solve(self_, **kwargs_):
                    other_truth_problem.solve(**kwargs_)
                    assign(self.truth_problem._solution, other_truth_problem._solution)
                    assign(self.truth_problem._solution_dot, other_truth_problem._solution_dot)
                    assign(self.truth_problem._solution_over_time, other_truth_problem._solution_over_time)
                    assign(self.truth_problem._solution_dot_over_time, other_truth_problem._solution_dot_over_time)
                    return self.truth_problem._solution_over_time

                self.patch_truth_solve = PatchInstanceMethod(
                    self.truth_problem,
                    "solve",
                    patched_truth_solve
                )
                self.patch_truth_solve.patch()

                # Initialize the affine expansion in the other truth problem
                other_truth_problem.init()
            else:
                other_truth_problem = self.truth_problem

            # Clean up solution caching and disable I/O
            if force:
                # Make sure to clean up problem and reduced problem solution cache to ensure that
                # solution and reduced solution are actually computed
                other_truth_problem._solution_over_time_cache.clear()
                other_truth_problem._solution_dot_over_time_cache.clear()
                self.reduced_problem._solution_over_time_cache.clear()
                self.reduced_problem._solution_dot_over_time_cache.clear()

                # Disable the capability of importing/exporting truth solutions
                def disable_import_solution_method(
                        self_, folder=None, filename=None, solution_over_time=None, component=None, suffix=None):
                    raise OSError

                self.disable_import_solution = PatchInstanceMethod(
                    other_truth_problem, "import_solution", disable_import_solution_method)
                self.disable_import_solution.patch()

                def disable_export_solution_method(
                        self_, folder=None, filename=None, solution_over_time=None, component=None, suffix=None):
                    pass

                self.disable_export_solution = PatchInstanceMethod(
                    other_truth_problem, "export_solution", disable_export_solution_method)
                self.disable_export_solution.patch()

        def _patch_truth_compute_output(self, force, **kwargs):
            if "with_respect_to" in kwargs:
                assert inspect.isfunction(kwargs["with_respect_to"])
                other_truth_problem = kwargs["with_respect_to"](self.truth_problem)

                def patched_truth_compute_output(self_):
                    other_truth_problem.compute_output()
                    assign(self.truth_problem._output, other_truth_problem._output)
                    assign(self.truth_problem._output_over_time, other_truth_problem._output_over_time)
                    return self.truth_problem._output_over_time

                self.patch_truth_compute_output = PatchInstanceMethod(
                    self.truth_problem,
                    "compute_output",
                    patched_truth_compute_output
                )
                self.patch_truth_compute_output.patch()

                # Initialize the affine expansion in the other truth problem
                other_truth_problem.init()
            else:
                other_truth_problem = self.truth_problem

            # Clean up output caching and disable I/O
            if force:
                # Make sure to clean up problem and reduced problem output cache to ensure that
                # output and reduced output are actually computed
                other_truth_problem._output_over_time_cache.clear()
                self.reduced_problem._output_over_time_cache.clear()

                # Disable the capability of importing/exporting truth output
                def disable_import_output_method(
                        self_, folder=None, filename=None, output_over_time=None, suffix=None):
                    raise OSError

                self.disable_import_output = PatchInstanceMethod(
                    other_truth_problem, "import_output", disable_import_output_method)
                self.disable_import_output.patch()

                def disable_export_output_method(
                        self_, folder=None, filename=None, output_over_time=None, suffix=None):
                    pass

                self.disable_export_output = PatchInstanceMethod(
                    other_truth_problem, "export_output", disable_export_output_method)
                self.disable_export_output.patch()

    # return value (a class) for the decorator
    return TimeDependentReductionMethod_Class
