# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from math import sqrt
from numbers import Number
from rbnics.backends import ProperOrthogonalDecomposition, TimeQuadrature
from rbnics.reduction_methods.base.pod_galerkin_reduction import PODGalerkinReduction
from rbnics.reduction_methods.base.time_dependent_reduction_method import TimeDependentReductionMethod
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators
from rbnics.utils.io import ErrorAnalysisTable


@RequiredBaseDecorators(PODGalerkinReduction, TimeDependentReductionMethod)
def TimeDependentPODGalerkinReduction(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class TimeDependentPODGalerkinReduction_Class(DifferentialProblemReductionMethod_DerivedClass):

        # Default initialization of members
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)

            # Compress first time trajectory (for each mu) and then parameter dependence
            # at the end of the offline stage, or carry out only one compression at the
            # end of the offline stage
            self.nested_POD = False  # by default only one compression
            # POD_time_trajectory: ProperOrthogonalDecomposition (for problems with one component)
            # or dict of ProperOrthogonalDecomposition (for problem with several components)
            self.POD_time_trajectory = None
            # Nested POD size
            self.N1 = 0
            # POD-Greedy tolerances. Since we use a POD for each component, it makes sense to possibly have
            # different tolerances for each component.
            if len(self.truth_problem.components) > 1:
                self.tol1 = {component: 0. for component in self.truth_problem.components}
            else:
                self.tol1 = 0.

        # OFFLINE: set maximum reduced space dimension (stopping criterion)
        def set_Nmax(self, Nmax, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            # Set nested POD sizes
            if "nested_POD" in kwargs:
                self.nested_POD = True
                self.N1 = kwargs["nested_POD"]

        # OFFLINE: set tolerance (stopping criterion)
        def set_tolerance(self, tol, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass.set_tolerance(self, tol, **kwargs)
            # Set nested POD tolerance
            if "nested_POD" in kwargs:
                assert self.nested_POD is True
                self.tol1 = self._preprocess_nested_POD_tolerance(kwargs["nested_POD"])

        def _preprocess_nested_POD_tolerance(self, tol):
            if len(self.truth_problem.components) > 1:
                assert isinstance(tol, (dict, Number))
                if isinstance(tol, dict):
                    for component in self.truth_problem.components:
                        assert component in tol, (
                            "You need to specify the tolerance of all components in tolerance dictionary")
                else:
                    tol_number = tol
                    tol = dict()
                    for component in self.truth_problem.components:
                        tol[component] = tol_number
            else:
                assert isinstance(tol, (dict, Number))
                if isinstance(tol, dict):
                    assert len(self.truth_problem.components) == 1
                    component_0 = self.truth_problem.components[0]
                    assert component_0 in tol
                    tol = tol[component_0]

            return tol

        # Initialize data structures required for the offline phase
        def _init_offline(self):
            # Call parent to initialize inner product
            output = DifferentialProblemReductionMethod_DerivedClass._init_offline(self)

            if self.nested_POD:
                # Declare new POD object(s)
                if len(self.truth_problem.components) > 1:
                    self.POD_time_trajectory = dict()
                    for component in self.truth_problem.components:
                        assert len(self.truth_problem.inner_product[component]) == 1
                        # the affine expansion storage contains only the inner product matrix
                        inner_product = self.truth_problem.inner_product[component][0]
                        self.POD_time_trajectory[component] = ProperOrthogonalDecomposition(
                            self.truth_problem.V, inner_product)
                else:
                    assert len(self.truth_problem.inner_product) == 1
                    # the affine expansion storage contains only the inner product matrix
                    inner_product = self.truth_problem.inner_product[0]
                    self.POD_time_trajectory = ProperOrthogonalDecomposition(
                        self.truth_problem.V, inner_product)

            # Return
            return output

        # Update the snapshots matrix
        def update_snapshots_matrix(self, snapshot_over_time):
            if self.nested_POD:
                if len(self.truth_problem.components) > 1:
                    for component in self.truth_problem.components:
                        (eigs1, basis_functions1) = self._nested_POD_compress_time_trajectory(
                            snapshot_over_time, component=component)
                        self.POD[component].store_snapshot(
                            basis_functions1, weight=[sqrt(e) for e in eigs1], component=component)
                else:
                    (eigs1, basis_functions1) = self._nested_POD_compress_time_trajectory(snapshot_over_time)
                    self.POD.store_snapshot(basis_functions1, weight=[sqrt(e) for e in eigs1])
            else:
                DifferentialProblemReductionMethod_DerivedClass.update_snapshots_matrix(self, snapshot_over_time)

        def _nested_POD_compress_time_trajectory(self, snapshot_over_time, component=None):
            N1 = self.N1
            if component is None:
                POD_time_trajectory = self.POD_time_trajectory
                tol1 = self.tol1
            else:
                POD_time_trajectory = self.POD_time_trajectory[component]
                tol1 = self.tol1[component]
            POD_time_trajectory.clear()
            POD_time_trajectory.store_snapshot(snapshot_over_time, component=component)
            (eigs1, _, basis_functions1, N1) = POD_time_trajectory.apply(N1, tol1)
            POD_time_trajectory.print_eigenvalues(N1)
            if component is None:
                POD_time_trajectory.save_eigenvalues_file(self.folder["post_processing"], "eigs")
                POD_time_trajectory.save_retained_energy_file(self.folder["post_processing"], "retained_energy")
            else:
                POD_time_trajectory.save_eigenvalues_file(
                    self.folder["post_processing"], "eigs_" + component)
                POD_time_trajectory.save_retained_energy_file(
                    self.folder["post_processing"], "retained_energy_" + component)
            return (eigs1, basis_functions1)

        # Compute the error of the reduced order approximation with respect to the full order one
        # over the testing set
        def error_analysis(self, N_generator=None, filename=None, **kwargs):
            if "components" in kwargs:
                components = kwargs["components"]
            else:
                components = self.truth_problem.components

            for component in components:
                def solution_preprocess_setitem(component):
                    def solution_preprocess_setitem__function(list_over_time):
                        list_squared_over_time = [v**2 for v in list_over_time]
                        time_quadrature = TimeQuadrature((0., self.truth_problem.T), list_squared_over_time)
                        return sqrt(time_quadrature.integrate())
                    return solution_preprocess_setitem__function
                for column_prefix in ("error_", "relative_error_"):
                    ErrorAnalysisTable.preprocess_setitem(
                        column_prefix + component, solution_preprocess_setitem(component))

            def output_preprocess_setitem(list_over_time):
                time_quadrature = TimeQuadrature((0., self.truth_problem.T), list_over_time)
                return time_quadrature.integrate()
            for column in ("error_output", "relative_error_output"):
                ErrorAnalysisTable.preprocess_setitem(column, solution_preprocess_setitem)

            DifferentialProblemReductionMethod_DerivedClass.error_analysis(self, N_generator, filename, **kwargs)

            ErrorAnalysisTable.clear_setitem_preprocessing()

    # return value (a class) for the decorator
    return TimeDependentPODGalerkinReduction_Class
