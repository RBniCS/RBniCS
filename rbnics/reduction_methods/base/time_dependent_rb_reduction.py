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

from math import sqrt
from numbers import Number
from logging import DEBUG, getLogger
from rbnics.backends import ProperOrthogonalDecomposition, SnapshotsMatrix, TimeQuadrature
from rbnics.reduction_methods.base.rb_reduction import RBReduction
from rbnics.reduction_methods.base.time_dependent_reduction_method import TimeDependentReductionMethod
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators
from rbnics.utils.io import ErrorAnalysisTable

logger = getLogger("rbnics/reduction_methods/base/time_dependent_rb_reduction.py")

@RequiredBaseDecorators(RBReduction, TimeDependentReductionMethod)
def TimeDependentRBReduction(DifferentialProblemReductionMethod_DerivedClass):
    
    @PreserveClassName
    class TimeDependentRBReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        
        # Default initialization of members
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # $$ OFFLINE DATA STRUCTURES $$ #
            # Choose among two versions of POD-Greedy
            self.POD_greedy_basis_extension = None # "orthogonal" or "POD"
            #   orthogonal ~> Haasdonk, Ohlberger; ESAIM: M2AN, 2008
            #   POD ~>  Nguyen, Rozza, Patera; Calcolo, 2009
            # Declare POD objects for basis computation using POD-Greedy
            self.POD_time_trajectory = None # ProperOrthogonalDecomposition (for problems with one component) or dict of ProperOrthogonalDecomposition (for problem with several components)
            self.POD_basis = None # ProperOrthogonalDecomposition (for problems with one component) or dict of ProperOrthogonalDecomposition (for problem with several components)
            # POD-Greedy size
            self.N1 = 0
            self.N2 = 0
            # POD-Greedy tolerances. Since we use a POD for each component, it makes sense to possibly have
            # different tolerances for each component.
            if len(self.truth_problem.components) > 1:
                self.tol1 = {component: 0. for component in self.truth_problem.components}
                self.tol2 = {component: 0. for component in self.truth_problem.components}
            else:
                self.tol1 = 0.
                self.tol2 = 0.
                
        # OFFLINE: set maximum reduced space dimension (stopping criterion)
        def set_Nmax(self, Nmax, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            # Set POD-Greedy sizes
            assert "POD_Greedy" in kwargs
            assert isinstance(kwargs["POD_Greedy"], (int, list, tuple))
            if isinstance(kwargs["POD_Greedy"], (list, tuple)):
                assert self.POD_greedy_basis_extension is None or self.POD_greedy_basis_extension == "POD"
                if self.POD_greedy_basis_extension is None:
                    self.POD_greedy_basis_extension = "POD"
                self.N1 = kwargs["POD_Greedy"][0]
                self.N2 = kwargs["POD_Greedy"][1]
                assert self.N1 > self.N2
            elif isinstance(kwargs["POD_Greedy"], int):
                assert self.POD_greedy_basis_extension is None or self.POD_greedy_basis_extension == "orthogonal"
                if self.POD_greedy_basis_extension is None:
                    self.POD_greedy_basis_extension = "orthogonal"
                self.N1 = kwargs["POD_Greedy"]
                
        # OFFLINE: set tolerance (stopping criterion)
        def set_tolerance(self, tol, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass.set_tolerance(self, tol, **kwargs)
            # Set POD-Greedy tolerance
            assert "POD_Greedy" in kwargs
            assert isinstance(kwargs["POD_Greedy"], (dict, Number, list, tuple))
            if isinstance(kwargs["POD_Greedy"], (list, tuple)):
                assert self.POD_greedy_basis_extension is None or self.POD_greedy_basis_extension == "POD"
                if self.POD_greedy_basis_extension is None:
                    self.POD_greedy_basis_extension = "POD"
                self.tol1 = self._preprocess_POD_greedy_tolerance(kwargs["POD_Greedy"][0])
                self.tol2 = self._preprocess_POD_greedy_tolerance(kwargs["POD_Greedy"][1])
                if len(self.truth_problem.components) > 1:
                    for component in self.truth_problem.components:
                        assert self.tol1[component] < self.tol2[component]
                else:
                    assert self.tol1 < self.tol2
            elif isinstance(kwargs["POD_Greedy"], (dict, Number)):
                assert self.POD_greedy_basis_extension is None or self.POD_greedy_basis_extension == "orthogonal"
                if self.POD_greedy_basis_extension is None:
                    self.POD_greedy_basis_extension = "orthogonal"
                self.tol1 = self._preprocess_POD_greedy_tolerance(kwargs["POD_Greedy"])
        
        def _preprocess_POD_greedy_tolerance(self, tol):
            if len(self.truth_problem.components) > 1:
                assert isinstance(tol, (dict, Number))
                if isinstance(tol, dict):
                    for component in self.truth_problem.components:
                        assert component in tol, "You need to specify the tolerance of all components in tolerance dictionary"
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
            
            # Check admissible values of POD_greedy_basis_extension
            assert self.POD_greedy_basis_extension in ("orthogonal", "POD")
            
            # Declare new POD object(s)
            if len(self.truth_problem.components) > 1:
                self.POD_time_trajectory = dict()
                if self.POD_greedy_basis_extension == "POD":
                    self.POD_basis = dict()
                for component in self.truth_problem.components:
                    assert len(self.truth_problem.inner_product[component]) == 1 # the affine expansion storage contains only the inner product matrix
                    inner_product = self.truth_problem.inner_product[component][0]
                    self.POD_time_trajectory[component] = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product)
                    if self.POD_greedy_basis_extension == "POD":
                        self.POD_basis[component] = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product)
            else:
                assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                inner_product = self.truth_problem.inner_product[0]
                self.POD_time_trajectory = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product)
                if self.POD_greedy_basis_extension == "POD":
                    self.POD_basis = ProperOrthogonalDecomposition(self.truth_problem.V, inner_product)
            
            # Return
            return output
            
        # Update basis matrix by POD-Greedy
        def update_basis_matrix(self, snapshot_over_time):
            if self.POD_greedy_basis_extension == "orthogonal":
                orthogonal_snapshot_over_time = self._POD_greedy_orthogonalize_snapshot(snapshot_over_time)
                if len(self.truth_problem.components) > 1:
                    for component in self.truth_problem.components:
                        print("# POD-Greedy for component", component)
                        (basis_functions1, N1) = self._POD_greedy_compute_basis_extension_with_orthogonal_snapshot(orthogonal_snapshot_over_time, component=component)
                        self.reduced_problem.basis_functions.enrich(basis_functions1, component=component)
                        self.reduced_problem.N[component] += N1
                else:
                    (basis_functions1, N1) = self._POD_greedy_compute_basis_extension_with_orthogonal_snapshot(orthogonal_snapshot_over_time)
                    self.reduced_problem.basis_functions.enrich(basis_functions1)
                    self.reduced_problem.N += N1
            elif self.POD_greedy_basis_extension == "POD":
                self.reduced_problem.basis_functions.clear()
                if len(self.truth_problem.components) > 1:
                    for component in self.truth_problem.components:
                        print("# POD-Greedy for component", component)
                        (basis_functions2, N_plus_N2) = self._POD_greedy_compute_basis_extension_with_POD(snapshot_over_time, component=component)
                        self.reduced_problem.basis_functions.enrich(basis_functions2, component=component)
                        self.reduced_problem.N[component] = N_plus_N2
                else:
                    (basis_functions2, N_plus_N2) = self._POD_greedy_compute_basis_extension_with_POD(snapshot_over_time)
                    self.reduced_problem.basis_functions.enrich(basis_functions2)
                    self.reduced_problem.N = N_plus_N2
                    
            self.reduced_problem.basis_functions.save(self.reduced_problem.folder["basis"], "basis")
                
        def _POD_greedy_orthogonalize_snapshot(self, snapshot_over_time):
            if self.reduced_problem.N > 0:
                basis_functions = self.reduced_problem.basis_functions
                projected_snapshot_N_over_time = self.reduced_problem.project(snapshot_over_time, on_dirichlet_bc=False)
                orthogonal_snapshot_over_time = SnapshotsMatrix(self.truth_problem.V)
                for (snapshot, projected_snapshot_N) in zip(snapshot_over_time, projected_snapshot_N_over_time):
                    orthogonal_snapshot_over_time.enrich(snapshot - basis_functions*projected_snapshot_N)
                return orthogonal_snapshot_over_time
            else:
                return snapshot_over_time
                
        def _POD_greedy_compute_basis_extension_with_orthogonal_snapshot(self, orthogonal_snapshot_over_time, component=None):
            N1 = self.N1
            if component is None:
                POD_time_trajectory = self.POD_time_trajectory
                tol1 = self.tol1
            else:
                POD_time_trajectory = self.POD_time_trajectory[component]
                tol1 = self.tol1[component]
            POD_time_trajectory.clear()
            POD_time_trajectory.store_snapshot(orthogonal_snapshot_over_time, component=component)
            (_, _, basis_functions1, N1) = POD_time_trajectory.apply(N1, tol1)
            POD_time_trajectory.print_eigenvalues(N1)
            if component is None:
                POD_time_trajectory.save_eigenvalues_file(self.folder["post_processing"], "eigs")
                POD_time_trajectory.save_retained_energy_file(self.folder["post_processing"], "retained_energy")
            else:
                POD_time_trajectory.save_eigenvalues_file(self.folder["post_processing"], "eigs_" + component)
                POD_time_trajectory.save_retained_energy_file(self.folder["post_processing"], "retained_energy_" + component)
            return (basis_functions1, N1)
            
        def _POD_greedy_compute_basis_extension_with_POD(self, snapshot_over_time, component=None):
            # First, compress the time trajectory stored in snapshot
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
            
            # Then, compress parameter dependence (thus, we do not clear the POD object)
            N2 = self.N2
            if component is None:
                POD_basis = self.POD_basis
                tol2 = self.tol2
            else:
                POD_basis = self.POD_basis[component]
                tol2 = self.tol2[component]
            POD_basis.store_snapshot(basis_functions1, weight=[sqrt(e) for e in eigs1], component=component)
            (_, _, basis_functions2, N_plus_N2) = POD_basis.apply(self.reduced_problem.N + N2, tol2)
            POD_basis.print_eigenvalues(N_plus_N2)
            if component is None:
                POD_basis.save_eigenvalues_file(self.folder["post_processing"], "eigs")
                POD_basis.save_retained_energy_file(self.folder["post_processing"], "retained_energy")
            else:
                POD_basis.save_eigenvalues_file(self.folder["post_processing"], "eigs_" + component)
                POD_basis.save_retained_energy_file(self.folder["post_processing"], "retained_energy_" + component)
            
            # Finally, we need to clear out previously computed Riesz representors for bilinear forms,
            # because POD-Greedy basis are not hierarchical from one greedy iteration to the next one
            for term in self.reduced_problem.riesz_terms:
                if self.reduced_problem.terms_order[term] > 1:
                    for q in range(self.reduced_problem.Q[term]):
                        self.reduced_problem.riesz[term][q].clear()
                
            # Return
            return (basis_functions2, N_plus_N2)
        
        # Choose the next parameter in the offline stage in a greedy fashion
        def _greedy(self):
            
            if self.reduced_problem.N > 0: # skip during initialization
                # Print some additional information related to the current value of the parameter
                error_over_time = self.reduced_problem.compute_error()
                error_squared_over_time = [v**2 for v in error_over_time]
                error_time_quadrature = TimeQuadrature((0., self.truth_problem.T), error_squared_over_time)
                error = sqrt(error_time_quadrature.integrate())
                print("absolute error for current mu =", error)
                error_estimator_over_time = self.reduced_problem.estimate_error()
                error_estimator_squared_over_time = [v**2 for v in error_estimator_over_time]
                error_estimator_time_quadrature = TimeQuadrature((0., self.truth_problem.T), error_estimator_squared_over_time)
                error_estimator = sqrt(error_estimator_time_quadrature.integrate())
                print("absolute error estimator for current mu =", error_estimator)
            
            # Carry out the actual greedy search
            def solve_and_estimate_error(mu):
                self.reduced_problem.set_mu(mu)
                self.reduced_problem.solve()
                error_estimator_over_time = self.reduced_problem.estimate_error()
                error_estimator_squared_over_time = [v**2 for v in error_estimator_over_time]
                time_quadrature = TimeQuadrature((0., self.truth_problem.T), error_estimator_squared_over_time)
                error_estimator = sqrt(time_quadrature.integrate())
                logger.log(DEBUG, "Error estimator for mu = " + str(mu) + " is " + str(error_estimator))
                return error_estimator
                
            if self.reduced_problem.N == 0:
                print("find initial mu")
            else:
                print("find next mu")
                
            return self.training_set.max(solve_and_estimate_error)
            
        # Compute the error of the reduced order approximation with respect to the full order one
        # over the testing set
        def error_analysis(self, N_generator=None, filename=None, **kwargs):
            if "components" in kwargs:
                components = kwargs["components"]
            else:
                components = self.truth_problem.components
            
            def solution_preprocess_setitem(list_over_time):
                list_squared_over_time = [v**2 for v in list_over_time]
                time_quadrature = TimeQuadrature((0., self.truth_problem.T), list_squared_over_time)
                return sqrt(time_quadrature.integrate())
                
            def output_preprocess_setitem(list_over_time):
                time_quadrature = TimeQuadrature((0., self.truth_problem.T), list_over_time)
                return time_quadrature.integrate()
            
            if len(components) > 1:
                all_components_string = ""
                for component in components:
                    all_components_string += component
                    for column_prefix in ("error_", "relative_error_"):
                        ErrorAnalysisTable.preprocess_setitem(column_prefix + component, solution_preprocess_setitem)
                for column_prefix in ("error_", "error_estimator_", "relative_error_", "relative_error_estimator_"):
                    ErrorAnalysisTable.preprocess_setitem(column_prefix + all_components_string, solution_preprocess_setitem)
            else:
                component = components[0]
                for column_prefix in ("error_", "error_estimator_", "relative_error_", "relative_error_estimator_"):
                    ErrorAnalysisTable.preprocess_setitem(column_prefix + component, solution_preprocess_setitem)
                
            for column in ("error_output", "error_estimator_output", "relative_error_output", "relative_error_estimator_output"):
                ErrorAnalysisTable.preprocess_setitem(column, solution_preprocess_setitem)
            
            DifferentialProblemReductionMethod_DerivedClass.error_analysis(self, N_generator, filename, **kwargs)
            
            ErrorAnalysisTable.clear_setitem_preprocessing()
        
    # return value (a class) for the decorator
    return TimeDependentRBReduction_Class
