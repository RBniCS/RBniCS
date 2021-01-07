# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from math import sqrt
from logging import DEBUG, getLogger
from rbnics.backends import GramSchmidt
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators, snapshot_links_to_cache
from rbnics.utils.io import (ErrorAnalysisTable, GreedySelectedParametersList, GreedyErrorEstimatorsList,
                             OnlineSizeDict, SpeedupAnalysisTable, TextBox, TextLine, Timer)

logger = getLogger("rbnics/reduction_methods/base/rb_reduction.py")


@RequiredBaseDecorators(None)
def RBReduction(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class RBReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        """
        The folders used to store the snapshots and for the post processing data, the parameters
        for the greedy algorithm and the error estimator evaluations are initialized.

        :param truth_problem: class of the truth problem to be solved.
        :return: reduced RB class.

        """

        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)

            # Declare a GS object
            # GramSchmidt (for problems with one component) or dict of GramSchmidt (for problem
            # with several components)
            self.GS = None
            # I/O
            self.folder["snapshots"] = os.path.join(self.folder_prefix, "snapshots")
            self.folder["post_processing"] = os.path.join(self.folder_prefix, "post_processing")
            self.greedy_selected_parameters = GreedySelectedParametersList()
            self.greedy_error_estimators = GreedyErrorEstimatorsList()
            self.label = "RB"

        def _init_offline(self):
            # Call parent to initialize inner product and reduced problem
            output = DifferentialProblemReductionMethod_DerivedClass._init_offline(self)

            # Declare a new GS for each basis component
            if len(self.truth_problem.components) > 1:
                self.GS = dict()
                for component in self.truth_problem.components:
                    assert len(self.truth_problem.inner_product[component]) == 1
                    inner_product = self.truth_problem.inner_product[component][0]
                    self.GS[component] = GramSchmidt(self.truth_problem.V, inner_product)
            else:
                assert len(self.truth_problem.inner_product) == 1
                inner_product = self.truth_problem.inner_product[0]
                self.GS = GramSchmidt(self.truth_problem.V, inner_product)

            # Return
            return output

        def offline(self):
            """
            It performs the offline phase of the reduced order model.

            :return: reduced_problem where all offline data are stored.
            """
            need_to_do_offline_stage = self._init_offline()
            if need_to_do_offline_stage:
                self._offline()
            self._finalize_offline()
            return self.reduced_problem

        @snapshot_links_to_cache
        def _offline(self):
            print(TextBox(self.truth_problem.name() + " " + self.label + " offline phase begins", fill="="))
            print("")

            # Initialize first parameter to be used
            self.reduced_problem.build_reduced_operators()
            self.reduced_problem.build_error_estimation_operators()
            (absolute_error_estimator_max, relative_error_estimator_max) = self.greedy()
            print("initial maximum absolute error estimator over training set =", absolute_error_estimator_max)
            print("initial maximum relative error estimator over training set =", relative_error_estimator_max)

            print("")

            iteration = 0
            while self.reduced_problem.N < self.Nmax and relative_error_estimator_max >= self.tol:
                print(TextLine("N = " + str(self.reduced_problem.N), fill="#"))

                print("truth solve for mu =", self.truth_problem.mu)
                snapshot = self.truth_problem.solve()
                self.truth_problem.export_solution(self.folder["snapshots"], "truth_" + str(iteration), snapshot)
                snapshot = self.postprocess_snapshot(snapshot, iteration)

                print("update basis matrix")
                self.update_basis_matrix(snapshot)
                iteration += 1

                print("build reduced operators")
                self.reduced_problem.build_reduced_operators()

                print("reduced order solve")
                self.reduced_problem.solve()

                print("build operators for error estimation")
                self.reduced_problem.build_error_estimation_operators()

                (absolute_error_estimator_max, relative_error_estimator_max) = self.greedy()
                print("maximum absolute error estimator over training set =", absolute_error_estimator_max)
                print("maximum relative error estimator over training set =", relative_error_estimator_max)

                print("")

            print(TextBox(self.truth_problem.name() + " " + self.label + " offline phase ends", fill="="))
            print("")

        def update_basis_matrix(self, snapshot):
            """
            It updates basis matrix.

            :param snapshot: last offline solution calculated.
            """
            if len(self.truth_problem.components) > 1:
                for component in self.truth_problem.components:
                    new_basis_function = self.GS[component].apply(
                        snapshot, self.reduced_problem.basis_functions[component][
                            self.reduced_problem.N_bc[component]:], component=component)
                    self.reduced_problem.basis_functions.enrich(new_basis_function, component=component)
                    self.reduced_problem.N[component] += 1
                self.reduced_problem.basis_functions.save(self.reduced_problem.folder["basis"], "basis")
            else:
                new_basis_function = self.GS.apply(snapshot, self.reduced_problem.basis_functions[
                    self.reduced_problem.N_bc:])
                self.reduced_problem.basis_functions.enrich(new_basis_function)
                self.reduced_problem.N += 1
                self.reduced_problem.basis_functions.save(self.reduced_problem.folder["basis"], "basis")

        def greedy(self):
            """
            It chooses the next parameter in the offline stage in a greedy fashion:
            wrapper with post processing of the result (in particular, set greedily selected parameter
            and save to file)

            :return: max error estimator and the comparison with the first one calculated.
            """
            (error_estimator_max, error_estimator_argmax) = self._greedy()
            self.truth_problem.set_mu(self.training_set[error_estimator_argmax])
            self.greedy_selected_parameters.append(self.training_set[error_estimator_argmax])
            self.greedy_selected_parameters.save(self.folder["post_processing"], "mu_greedy")
            self.greedy_error_estimators.append(error_estimator_max)
            self.greedy_error_estimators.save(self.folder["post_processing"], "error_estimator_max")
            return (error_estimator_max, error_estimator_max / self.greedy_error_estimators[0])

        def _greedy(self):
            """
            It chooses the next parameter in the offline stage in a greedy fashion. Internal method.

            :return: max error estimator and the respective parameter.
            """

            if self.reduced_problem.N > 0:  # skip during initialization
                # Print some additional information on the consistency of the reduced basis
                print("absolute error for current mu =", self.reduced_problem.compute_error())
                print("absolute error estimator for current mu =", self.reduced_problem.estimate_error())

            # Carry out the actual greedy search
            def solve_and_estimate_error(mu):
                self.reduced_problem.set_mu(mu)
                self.reduced_problem.solve()
                error_estimator = self.reduced_problem.estimate_error()
                logger.log(DEBUG, "Error estimator for mu = " + str(mu) + " is " + str(error_estimator))
                return error_estimator

            if self.reduced_problem.N == 0:
                print("find initial mu")
            else:
                print("find next mu")

            return self.training_set.max(solve_and_estimate_error)

        def error_analysis(self, N_generator=None, filename=None, **kwargs):
            """
            It computes the error of the reduced order approximation with respect to the full order one
            over the testing set.

            :param N_generator: generator of dimension of reduced problem.
            """
            self._init_error_analysis(**kwargs)
            self._error_analysis(N_generator, filename, **kwargs)
            self._finalize_error_analysis(**kwargs)

        def _error_analysis(self, N_generator=None, filename=None, **kwargs):
            if N_generator is None:
                def N_generator():
                    N = self.reduced_problem.N
                    if isinstance(N, dict):
                        N = min(N.values())
                    for n in range(1, N + 1):  # n = 1, ... N
                        yield n

            if "components" in kwargs:
                components = kwargs["components"]
            else:
                components = self.truth_problem.components

            def N_generator_items():
                for n in N_generator():
                    assert isinstance(n, (dict, int))
                    if isinstance(n, int):
                        yield (n, n)
                    elif isinstance(n, dict):
                        assert len(n) == 1
                        (n_int, n_online_size_dict) = n.popitem()
                        assert isinstance(n_int, int)
                        assert isinstance(n_online_size_dict, OnlineSizeDict)
                        yield (n_int, n_online_size_dict)
                    else:
                        raise TypeError("Invalid item generated by N_generator")

            def N_generator_max():
                *_, Nmax = N_generator_items()
                assert isinstance(Nmax, tuple)
                assert len(Nmax) == 2
                assert isinstance(Nmax[0], int)
                return Nmax[0]

            print(TextBox(self.truth_problem.name() + " " + self.label + " error analysis begins", fill="="))
            print("")

            error_analysis_table = ErrorAnalysisTable(self.testing_set)
            error_analysis_table.set_Nmax(N_generator_max())
            if len(components) > 1:
                all_components_string = "".join(components)
                for component in components:
                    error_analysis_table.add_column(
                        "error_" + component,
                        group_name="solution_" + component + "_error",
                        operations=("mean", "max"))
                    error_analysis_table.add_column(
                        "relative_error_" + component,
                        group_name="solution_" + component + "_relative_error",
                        operations=("mean", "max"))
                error_analysis_table.add_column(
                    "error_" + all_components_string,
                    group_name="solution_" + all_components_string + "_error",
                    operations=("mean", "max"))
                error_analysis_table.add_column(
                    "error_estimator_" + all_components_string,
                    group_name="solution_" + all_components_string + "_error",
                    operations=("mean", "max"))
                error_analysis_table.add_column(
                    "effectivity_" + all_components_string,
                    group_name="solution_" + all_components_string + "_error",
                    operations=("min", "mean", "max"))
                error_analysis_table.add_column(
                    "relative_error_" + all_components_string,
                    group_name="solution_" + all_components_string + "_relative_error",
                    operations=("mean", "max"))
                error_analysis_table.add_column(
                    "relative_error_estimator_" + all_components_string,
                    group_name="solution_" + all_components_string + "_relative_error",
                    operations=("mean", "max"))
                error_analysis_table.add_column(
                    "relative_effectivity_" + all_components_string,
                    group_name="solution_" + all_components_string + "_relative_error",
                    operations=("min", "mean", "max"))
            else:
                component = components[0]
                error_analysis_table.add_column(
                    "error_" + component,
                    group_name="solution_" + component + "_error",
                    operations=("mean", "max"))
                error_analysis_table.add_column(
                    "error_estimator_" + component,
                    group_name="solution_" + component + "_error",
                    operations=("mean", "max"))
                error_analysis_table.add_column(
                    "effectivity_" + component,
                    group_name="solution_" + component + "_error",
                    operations=("min", "mean", "max"))
                error_analysis_table.add_column(
                    "relative_error_" + component,
                    group_name="solution_" + component + "_relative_error",
                    operations=("mean", "max"))
                error_analysis_table.add_column(
                    "relative_error_estimator_" + component,
                    group_name="solution_" + component + "_relative_error",
                    operations=("mean", "max"))
                error_analysis_table.add_column(
                    "relative_effectivity_" + component,
                    group_name="solution_" + component + "_relative_error",
                    operations=("min", "mean", "max"))
            error_analysis_table.add_column(
                "error_output", group_name="output_error", operations=("mean", "max"))
            error_analysis_table.add_column(
                "error_estimator_output", group_name="output_error", operations=("mean", "max"))
            error_analysis_table.add_column(
                "effectivity_output", group_name="output_error", operations=("min", "mean", "max"))
            error_analysis_table.add_column(
                "relative_error_output", group_name="output_relative_error", operations=("mean", "max"))
            error_analysis_table.add_column(
                "relative_error_estimator_output", group_name="output_relative_error", operations=("mean", "max"))
            error_analysis_table.add_column(
                "relative_effectivity_output", group_name="output_relative_error", operations=("min", "mean", "max"))

            for (mu_index, mu) in enumerate(self.testing_set):
                print(TextLine(str(mu_index), fill="#"))

                self.reduced_problem.set_mu(mu)

                for (n_int, n_arg) in N_generator_items():
                    self.reduced_problem.solve(n_arg, **kwargs)
                    error = self.reduced_problem.compute_error(**kwargs)
                    if len(components) > 1:
                        error[all_components_string] = sqrt(
                            sum([error[component]**2 for component in components]))
                    error_estimator = self.reduced_problem.estimate_error()
                    relative_error = self.reduced_problem.compute_relative_error(**kwargs)
                    if len(components) > 1:
                        relative_error[all_components_string] = sqrt(
                            sum([relative_error[component]**2 for component in components]))
                    relative_error_estimator = self.reduced_problem.estimate_relative_error()

                    self.reduced_problem.compute_output()
                    error_output = self.reduced_problem.compute_error_output(**kwargs)
                    error_output_estimator = self.reduced_problem.estimate_error_output()
                    relative_error_output = self.reduced_problem.compute_relative_error_output(**kwargs)
                    relative_error_output_estimator = self.reduced_problem.estimate_relative_error_output()

                    if len(components) > 1:
                        for component in components:
                            error_analysis_table[
                                "error_" + component, n_int, mu_index] = error[component]
                            error_analysis_table[
                                "relative_error_" + component, n_int, mu_index] = relative_error[component]
                        error_analysis_table[
                            "error_" + all_components_string, n_int, mu_index] = error[all_components_string]
                        error_analysis_table[
                            "error_estimator_" + all_components_string, n_int, mu_index] = error_estimator
                        error_analysis_table[
                            "effectivity_" + all_components_string, n_int, mu_index] = error_analysis_table[
                                "error_estimator_" + all_components_string, n_int, mu_index] / error_analysis_table[
                                    "error_" + all_components_string, n_int, mu_index]
                        error_analysis_table[
                            "relative_error_" + all_components_string, n_int, mu_index] = relative_error[
                                all_components_string]
                        error_analysis_table[
                            "relative_error_estimator_" + all_components_string, n_int,
                            mu_index] = relative_error_estimator
                        error_analysis_table[
                            "relative_effectivity_" + all_components_string, n_int,
                            mu_index] = error_analysis_table[
                                "relative_error_estimator_" + all_components_string, n_int,
                                mu_index] / error_analysis_table[
                                    "relative_error_" + all_components_string, n_int, mu_index]
                    else:
                        component = components[0]
                        error_analysis_table["error_" + component, n_int, mu_index] = error
                        error_analysis_table["error_estimator_" + component, n_int, mu_index] = error_estimator
                        error_analysis_table[
                            "effectivity_" + component, n_int, mu_index] = error_analysis_table[
                                "error_estimator_" + component, n_int, mu_index] / error_analysis_table[
                                    "error_" + component, n_int, mu_index]
                        error_analysis_table["relative_error_" + component, n_int, mu_index] = relative_error
                        error_analysis_table[
                            "relative_error_estimator_" + component, n_int, mu_index] = relative_error_estimator
                        error_analysis_table[
                            "relative_effectivity_" + component, n_int, mu_index] = error_analysis_table[
                                "relative_error_estimator_" + component, n_int, mu_index] / error_analysis_table[
                                    "relative_error_" + component, n_int, mu_index]

                    error_analysis_table["error_output", n_int, mu_index] = error_output
                    error_analysis_table["error_estimator_output", n_int, mu_index] = error_output_estimator
                    error_analysis_table[
                        "effectivity_output", n_int, mu_index] = error_analysis_table[
                            "error_estimator_output", n_int, mu_index] / error_analysis_table[
                                "error_output", n_int, mu_index]
                    error_analysis_table["relative_error_output", n_int, mu_index] = relative_error_output
                    error_analysis_table[
                        "relative_error_estimator_output", n_int, mu_index] = relative_error_output_estimator
                    error_analysis_table[
                        "relative_effectivity_output", n_int, mu_index] = error_analysis_table[
                            "relative_error_estimator_output", n_int, mu_index] / error_analysis_table[
                                "relative_error_output", n_int, mu_index]

            # Print
            print("")
            print(error_analysis_table)

            print("")
            print(TextBox(self.truth_problem.name() + " " + self.label + " error analysis ends", fill="="))
            print("")

            # Export error analysis table
            error_analysis_table.save(
                self.folder["error_analysis"], "error_analysis" if filename is None else filename)

        def speedup_analysis(self, N_generator=None, filename=None, **kwargs):
            """
            It computes the speedup of the reduced order approximation with respect to the full order one
            over the testing set.

            :param N_generator: generator of dimension of the reduced problem.
            """
            self._init_speedup_analysis(**kwargs)
            self._speedup_analysis(N_generator, filename, **kwargs)
            self._finalize_speedup_analysis(**kwargs)

        def _speedup_analysis(self, N_generator=None, filename=None, **kwargs):
            if N_generator is None:
                def N_generator():
                    N = self.reduced_problem.N
                    if isinstance(N, dict):
                        N = min(N.values())
                    for n in range(1, N + 1):  # n = 1, ... N
                        yield n

            def N_generator_items():
                for n in N_generator():
                    assert isinstance(n, (dict, int))
                    if isinstance(n, int):
                        yield (n, n)
                    elif isinstance(n, dict):
                        assert len(n) == 1
                        (n_int, n_online_size_dict) = n.popitem()
                        assert isinstance(n_int, int)
                        assert isinstance(n_online_size_dict, OnlineSizeDict)
                        yield (n_int, n_online_size_dict)
                    else:
                        raise TypeError("Invalid item generated by N_generator")

            def N_generator_max():
                *_, Nmax = N_generator_items()
                assert isinstance(Nmax, tuple)
                assert len(Nmax) == 2
                assert isinstance(Nmax[0], int)
                return Nmax[0]

            print(TextBox(self.truth_problem.name() + " " + self.label + " speedup analysis begins", fill="="))
            print("")

            speedup_analysis_table = SpeedupAnalysisTable(self.testing_set)
            speedup_analysis_table.set_Nmax(N_generator_max())
            speedup_analysis_table.add_column(
                "speedup_solve",
                group_name="speedup_solve",
                operations=("min", "mean", "max"))
            speedup_analysis_table.add_column(
                "speedup_solve_and_estimate_error",
                group_name="speedup_solve_and_estimate_error",
                operations=("min", "mean", "max"))
            speedup_analysis_table.add_column(
                "speedup_solve_and_estimate_relative_error",
                group_name="speedup_solve_and_estimate_relative_error",
                operations=("min", "mean", "max"))
            speedup_analysis_table.add_column(
                "speedup_output",
                group_name="speedup_output",
                operations=("min", "mean", "max"))
            speedup_analysis_table.add_column(
                "speedup_output_and_estimate_error_output",
                group_name="speedup_output_and_estimate_error_output",
                operations=("min", "mean", "max"))
            speedup_analysis_table.add_column(
                "speedup_output_and_estimate_relative_error_output",
                group_name="speedup_output_and_estimate_relative_error_output",
                operations=("min", "mean", "max"))

            truth_timer = Timer("parallel")
            reduced_timer = Timer("serial")

            for (mu_index, mu) in enumerate(self.testing_set):
                print(TextLine(str(mu_index), fill="#"))

                self.reduced_problem.set_mu(mu)

                truth_timer.start()
                self.truth_problem.solve(**kwargs)
                elapsed_truth_solve = truth_timer.stop()

                truth_timer.start()
                self.truth_problem.compute_output()
                elapsed_truth_output = truth_timer.stop()

                for (n_int, n_arg) in N_generator_items():
                    reduced_timer.start()
                    solution = self.reduced_problem.solve(n_arg, **kwargs)
                    elapsed_reduced_solve = reduced_timer.stop()

                    truth_timer.start()
                    self.reduced_problem.compute_error(**kwargs)
                    elapsed_error = truth_timer.stop()

                    reduced_timer.start()
                    error_estimator = self.reduced_problem.estimate_error()
                    elapsed_error_estimator = reduced_timer.stop()

                    truth_timer.start()
                    self.reduced_problem.compute_relative_error(**kwargs)
                    elapsed_relative_error = truth_timer.stop()

                    reduced_timer.start()
                    relative_error_estimator = self.reduced_problem.estimate_relative_error()
                    elapsed_relative_error_estimator = reduced_timer.stop()

                    reduced_timer.start()
                    output = self.reduced_problem.compute_output()
                    elapsed_reduced_output = reduced_timer.stop()

                    truth_timer.start()
                    self.reduced_problem.compute_error_output(**kwargs)
                    elapsed_error_output = truth_timer.stop()

                    reduced_timer.start()
                    error_estimator_output = self.reduced_problem.estimate_error_output()
                    elapsed_error_estimator_output = reduced_timer.stop()

                    truth_timer.start()
                    self.reduced_problem.compute_relative_error_output(**kwargs)
                    elapsed_relative_error_output = truth_timer.stop()

                    reduced_timer.start()
                    relative_error_estimator_output = self.reduced_problem.estimate_relative_error_output()
                    elapsed_relative_error_estimator_output = reduced_timer.stop()

                    if solution is not NotImplemented:
                        speedup_analysis_table[
                            "speedup_solve", n_int, mu_index] = elapsed_truth_solve / elapsed_reduced_solve
                    else:
                        speedup_analysis_table[
                            "speedup_solve", n_int, mu_index] = NotImplemented
                    if error_estimator is not NotImplemented:
                        speedup_analysis_table[
                            "speedup_solve_and_estimate_error", n_int, mu_index] = (
                                elapsed_truth_solve + elapsed_error) / (
                                    elapsed_reduced_solve + elapsed_error_estimator)
                    else:
                        speedup_analysis_table[
                            "speedup_solve_and_estimate_error", n_int, mu_index] = NotImplemented
                    if relative_error_estimator is not NotImplemented:
                        speedup_analysis_table[
                            "speedup_solve_and_estimate_relative_error", n_int, mu_index] = (
                                elapsed_truth_solve + elapsed_relative_error) / (
                                    elapsed_reduced_solve + elapsed_relative_error_estimator)
                    else:
                        speedup_analysis_table[
                            "speedup_solve_and_estimate_relative_error", n_int, mu_index] = NotImplemented
                    if output is not NotImplemented:
                        speedup_analysis_table[
                            "speedup_output", n_int, mu_index] = (
                                elapsed_truth_solve + elapsed_truth_output) / (
                                    elapsed_reduced_solve + elapsed_reduced_output)
                    else:
                        speedup_analysis_table[
                            "speedup_output", n_int, mu_index] = NotImplemented
                    if error_estimator_output is not NotImplemented:
                        assert output is not NotImplemented
                        speedup_analysis_table[
                            "speedup_output_and_estimate_error_output", n_int, mu_index] = (
                                elapsed_truth_solve + elapsed_truth_output + elapsed_error_output) / (
                                    elapsed_reduced_solve + elapsed_reduced_output + elapsed_error_estimator_output)
                    else:
                        speedup_analysis_table[
                            "speedup_output_and_estimate_error_output", n_int, mu_index] = NotImplemented
                    if relative_error_estimator_output is not NotImplemented:
                        assert output is not NotImplemented
                        speedup_analysis_table[
                            "speedup_output_and_estimate_relative_error_output", n_int, mu_index] = (
                                elapsed_truth_solve + elapsed_truth_output + elapsed_relative_error_output) / (
                                    elapsed_reduced_solve + elapsed_reduced_output
                                    + elapsed_relative_error_estimator_output)
                    else:
                        speedup_analysis_table[
                            "speedup_output_and_estimate_relative_error_output", n_int, mu_index] = NotImplemented

            # Print
            print("")
            print(speedup_analysis_table)

            print("")
            print(TextBox(self.truth_problem.name() + " " + self.label + " speedup analysis ends", fill="="))
            print("")

            # Export speedup analysis table
            speedup_analysis_table.save(
                self.folder["speedup_analysis"], "speedup_analysis" if filename is None else filename)

    # return value (a class) for the decorator
    return RBReduction_Class
