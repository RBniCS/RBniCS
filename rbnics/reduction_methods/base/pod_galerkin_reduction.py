# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from numbers import Number
from rbnics.backends import ProperOrthogonalDecomposition
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators, snapshot_links_to_cache
from rbnics.utils.io import ErrorAnalysisTable, OnlineSizeDict, SpeedupAnalysisTable, TextBox, TextLine, Timer


@RequiredBaseDecorators(None)
def PODGalerkinReduction(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class PODGalerkinReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        """
        Abstract class. The folders used to store the snapshots and for the post processing data,
        the data stracture for the POD algorithm are initialized.

        :param truth_problem: the class of the truth problem to be solved.
        :return: PODGalerkinReduction_Class where all the offline data are stored.
        """

        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)

            # Declare a POD object
            # ProperOrthogonalDecomposition (for problems with one component)
            # or dict of ProperOrthogonalDecomposition (for problem with several components)
            self.POD = None
            # I/O
            self.folder["snapshots"] = os.path.join(self.folder_prefix, "snapshots")
            self.folder["post_processing"] = os.path.join(self.folder_prefix, "post_processing")
            self.label = "POD-Galerkin"

            # Since we use a POD for each component, it makes sense to possibly have
            # different tolerances for each component.
            if len(self.truth_problem.components) > 1:
                self.tol = {component: 0. for component in self.truth_problem.components}
            else:
                self.tol = 0.

        def set_tolerance(self, tol, **kwargs):
            """
            It sets tolerance to be used as stopping criterion.

            :param tol: the tolerance to be used.
            """
            if len(self.truth_problem.components) > 1:
                if tol is None:
                    all_components_in_kwargs = self.truth_problem.components[0] in kwargs
                    for component in self.truth_problem.components:
                        if all_components_in_kwargs:
                            assert component in kwargs, (
                                "You need to specify the tolerance of all components in kwargs")
                        else:
                            assert component not in kwargs, (
                                "You need to specify the tolerance of all components in kwargs")
                    assert all_components_in_kwargs
                    tol = dict()
                    for component in self.truth_problem.components:
                        tol[component] = kwargs[component]
                        del kwargs[component]
                else:
                    assert isinstance(tol, Number)
                    tol_number = tol
                    tol = dict()
                    for component in self.truth_problem.components:
                        tol[component] = tol_number
                        assert component not in kwargs, (
                            "You cannot provide both a number and kwargs for components")
            else:
                if tol is None:
                    assert len(self.truth_problem.components) == 1
                    component_0 = self.truth_problem.components[0]
                    assert component_0 in kwargs
                    tol = kwargs[component_0]
                else:
                    assert isinstance(tol, Number)

            self.tol = tol

        def _init_offline(self):
            # Call parent to initialize inner product and reduced problem
            output = DifferentialProblemReductionMethod_DerivedClass._init_offline(self)

            # Declare a new POD for each basis component
            if len(self.truth_problem.components) > 1:
                self.POD = dict()
                for component in self.truth_problem.components:
                    assert len(self.truth_problem.inner_product[component]) == 1
                    # the affine expansion storage contains only the inner product matrix
                    inner_product = self.truth_problem.inner_product[component][0]
                    self.POD[component] = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product)
            else:
                assert len(self.truth_problem.inner_product) == 1
                # the affine expansion storage contains only the inner product matrix
                inner_product = self.truth_problem.inner_product[0]
                self.POD = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product)

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

            for (mu_index, mu) in enumerate(self.training_set):
                print(TextLine(str(mu_index), fill="#"))

                self.truth_problem.set_mu(mu)

                print("truth solve for mu =", self.truth_problem.mu)
                snapshot = self.truth_problem.solve()
                self.truth_problem.export_solution(self.folder["snapshots"], "truth_" + str(mu_index), snapshot)
                snapshot = self.postprocess_snapshot(snapshot, mu_index)

                print("update snapshots matrix")
                self.update_snapshots_matrix(snapshot)

                print("")

            print(TextLine("perform POD", fill="#"))
            self.compute_basis_functions()

            print("")
            print("build reduced operators")
            self.reduced_problem.build_reduced_operators()

            print("")
            print(TextBox(self.truth_problem.name() + " " + self.label + " offline phase ends", fill="="))
            print("")

        def update_snapshots_matrix(self, snapshot):
            """
            It updates the snapshots matrix.

            :param snapshot: last offline solution computed.
            """
            if len(self.truth_problem.components) > 1:
                for component in self.truth_problem.components:
                    self.POD[component].store_snapshot(snapshot, component=component)
            else:
                self.POD.store_snapshot(snapshot)

        def compute_basis_functions(self):
            """
            It computes basis functions performing POD solving an eigenvalue problem.
            """
            if len(self.truth_problem.components) > 1:
                for component in self.truth_problem.components:
                    print("# POD for component", component)
                    POD = self.POD[component]
                    (_, _, basis_functions, N) = POD.apply(self.Nmax, self.tol[component])
                    self.reduced_problem.basis_functions.enrich(basis_functions, component=component)
                    self.reduced_problem.N[component] += N
                    POD.print_eigenvalues(N)
                    POD.save_eigenvalues_file(self.folder["post_processing"], "eigs_" + component)
                    POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy_" + component)
                self.reduced_problem.basis_functions.save(self.reduced_problem.folder["basis"], "basis")
            else:
                (_, _, basis_functions, N) = self.POD.apply(self.Nmax, self.tol)
                self.reduced_problem.basis_functions.enrich(basis_functions)
                self.reduced_problem.N += N
                self.POD.print_eigenvalues(N)
                self.POD.save_eigenvalues_file(self.folder["post_processing"], "eigs")
                self.POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy")
                self.reduced_problem.basis_functions.save(self.reduced_problem.folder["basis"], "basis")

        def error_analysis(self, N_generator=None, filename=None, **kwargs):
            """
            It computes the error of the reduced order approximation with respect to the full order one
            over the testing set

            :param N_generator: generator of dimension of the reduced problem.
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
            for component in components:
                error_analysis_table.add_column(
                    "error_" + component, group_name="solution_" + component, operations=("mean", "max"))
                error_analysis_table.add_column(
                    "relative_error_" + component, group_name="solution_" + component, operations=("mean", "max"))
            error_analysis_table.add_column("error_output", group_name="output", operations=("mean", "max"))
            error_analysis_table.add_column("relative_error_output", group_name="output", operations=("mean", "max"))

            for (mu_index, mu) in enumerate(self.testing_set):
                print(TextLine(str(mu_index), fill="#"))

                self.reduced_problem.set_mu(mu)

                for (n_int, n_arg) in N_generator_items():
                    self.reduced_problem.solve(n_arg, **kwargs)
                    error = self.reduced_problem.compute_error(**kwargs)
                    relative_error = self.reduced_problem.compute_relative_error(**kwargs)

                    self.reduced_problem.compute_output()
                    error_output = self.reduced_problem.compute_error_output(**kwargs)
                    relative_error_output = self.reduced_problem.compute_relative_error_output(**kwargs)

                    if len(components) > 1:
                        for component in components:
                            error_analysis_table["error_" + component, n_int, mu_index] = error[component]
                            error_analysis_table[
                                "relative_error_" + component, n_int, mu_index] = relative_error[component]
                    else:
                        component = components[0]
                        error_analysis_table["error_" + component, n_int, mu_index] = error
                        error_analysis_table["relative_error_" + component, n_int, mu_index] = relative_error

                    error_analysis_table["error_output", n_int, mu_index] = error_output
                    error_analysis_table["relative_error_output", n_int, mu_index] = relative_error_output

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
            over the testing set

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
                "speedup_solve", group_name="speedup_solve", operations=("min", "mean", "max"))
            speedup_analysis_table.add_column(
                "speedup_output", group_name="speedup_output", operations=("min", "mean", "max"))

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

                    reduced_timer.start()
                    output = self.reduced_problem.compute_output()
                    elapsed_reduced_output = reduced_timer.stop()

                    if solution is not NotImplemented:
                        speedup_analysis_table[
                            "speedup_solve", n_int, mu_index] = elapsed_truth_solve / elapsed_reduced_solve
                    else:
                        speedup_analysis_table["speedup_solve", n_int, mu_index] = NotImplemented
                    if output is not NotImplemented:
                        speedup_analysis_table[
                            "speedup_output", n_int, mu_index] = (
                                elapsed_truth_solve + elapsed_truth_output) / (
                                    elapsed_reduced_solve + elapsed_reduced_output)
                    else:
                        speedup_analysis_table["speedup_output", n_int, mu_index] = NotImplemented

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
    return PODGalerkinReduction_Class
