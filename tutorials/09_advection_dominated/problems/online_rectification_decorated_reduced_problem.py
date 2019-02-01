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

from itertools import product as cartesian_product
from numpy.linalg import cond
from rbnics.backends import LinearSolver, SnapshotsMatrix, transpose
from rbnics.backends.online import OnlineAffineExpansionStorage, OnlineFunction
from rbnics.utils.decorators import PreserveClassName, ReducedProblemDecoratorFor
from rbnics.utils.io import GreedySelectedParametersList
from backends.online import OnlineMatrix, OnlineNonHierarchicalAffineExpansionStorage, OnlineSolveKwargsGenerator
from .online_rectification import OnlineRectification

@ReducedProblemDecoratorFor(OnlineRectification)
def OnlineRectificationDecoratedReducedProblem(EllipticCoerciveReducedProblem_DerivedClass):
    
    @PreserveClassName
    class OnlineRectificationDecoratedReducedProblem_Class(EllipticCoerciveReducedProblem_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            EllipticCoerciveReducedProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Copy of greedy snapshots
            self.snapshots_mu = GreedySelectedParametersList() # the difference between this list and greedy_selected_parameters in the reduction method is that this one also stores the initial parameter
            self.snapshots = SnapshotsMatrix(truth_problem.V)
            
            # Extend allowed keywords argument in solve
            self._online_solve_default_kwargs["online_rectification"] = True
            self.OnlineSolveKwargs = OnlineSolveKwargsGenerator(**self._online_solve_default_kwargs)
            
            # Generate all combinations of allowed keyword arguments in solve
            online_solve_kwargs_with_rectification = list()
            online_solve_kwargs_without_rectification = list()
            for other_args in cartesian_product((True, False), repeat=len(self._online_solve_default_kwargs) - 1):
                args_with_rectification = self.OnlineSolveKwargs(*(other_args + (True, )))
                args_without_rectification = self.OnlineSolveKwargs(*(other_args + (False, )))
                online_solve_kwargs_with_rectification.append(args_with_rectification)
                online_solve_kwargs_without_rectification.append(args_without_rectification)
            self.online_solve_kwargs_with_rectification = online_solve_kwargs_with_rectification
            self.online_solve_kwargs_without_rectification = online_solve_kwargs_without_rectification
            
            # Flag to disable error estimation after rectification has been setup
            self._disable_error_estimation = False
            
        def _init_operators(self, current_stage="online"):
            # Initialize additional reduced operators related to rectification. Note that these operators
            # are not hierarchical because:
            # * the basis is possibly non-hierarchical
            # * the coefficients of the reduced solution for different reduced sizes are definitely not hierarchical
            if current_stage == "online":
                self.operator["projection_truth_snapshots"] = OnlineNonHierarchicalAffineExpansionStorage(1)
                assert len(self.online_solve_kwargs_with_rectification) == len(self.online_solve_kwargs_without_rectification)
                self.operator["projection_reduced_snapshots"] = OnlineNonHierarchicalAffineExpansionStorage(len(self.online_solve_kwargs_with_rectification))
                self.assemble_operator("projection_truth_snapshots", "online")
                self.assemble_operator("projection_reduced_snapshots", "online")
                # Call Parent
                EllipticCoerciveReducedProblem_DerivedClass._init_operators(self, current_stage)
            elif current_stage == "offline":
                # Call Parent
                EllipticCoerciveReducedProblem_DerivedClass._init_operators(self, current_stage)
            elif current_stage == "offline_rectification_postprocessing":
                self.operator["projection_truth_snapshots"] = OnlineNonHierarchicalAffineExpansionStorage(1)
                assert len(self.online_solve_kwargs_with_rectification) == len(self.online_solve_kwargs_without_rectification)
                self.operator["projection_reduced_snapshots"] = OnlineNonHierarchicalAffineExpansionStorage(len(self.online_solve_kwargs_with_rectification))
                # We do not call Parent method as there is no need to re-initialize offline operators
            else:
                # Call Parent, which may eventually raise an error
                EllipticCoerciveReducedProblem_DerivedClass._init_operators(self, current_stage)
                
        def _init_inner_products(self, current_stage="online"):
            if current_stage in ("online", "offline"):
                # Call Parent
                EllipticCoerciveReducedProblem_DerivedClass._init_inner_products(self, current_stage)
            elif current_stage == "offline_rectification_postprocessing":
                pass # We do not call Parent method as there is no need to re-initialize offline inner products
            else:
                # Call Parent, which may eventually raise an error
                EllipticCoerciveReducedProblem_DerivedClass._init_inner_products(self, current_stage)
                
        def _init_basis_functions(self, current_stage="online"):
            if current_stage in ("online", "offline"):
                # Call Parent
                EllipticCoerciveReducedProblem_DerivedClass._init_basis_functions(self, current_stage)
            elif current_stage == "offline_rectification_postprocessing":
                pass # We do not call Parent method as there is no need to re-initialize offline basis functions
            else:
                # Call Parent, which may eventually raise an error
                EllipticCoerciveReducedProblem_DerivedClass._init_basis_functions(self, current_stage)
                
        def _init_error_estimation_operators(self, current_stage="online"):
            if current_stage in ("online", "offline_rectification_postprocessing"):
                # Disable error estimation, which would not take into account the additional rectification
                self._disable_error_estimation = True
            elif current_stage == "offline":
                # Call Parent
                EllipticCoerciveReducedProblem_DerivedClass._init_error_estimation_operators(self, current_stage)
            else:
                # Call Parent, which may eventually raise an error
                EllipticCoerciveReducedProblem_DerivedClass._init_error_estimation_operators(self, current_stage)
            
        def build_reduced_operators(self, current_stage="offline"):
            if current_stage == "offline_rectification_postprocessing":
                # Compute projection of truth and reduced snapshots
                print("build projection truth snapshots for rectification")
                self.operator["projection_truth_snapshots"] = self.assemble_operator("projection_truth_snapshots", "offline_rectification_postprocessing")
                print("build projection reduced snapshots for rectification")
                self.operator["projection_reduced_snapshots"] = self.assemble_operator("projection_reduced_snapshots", "offline_rectification_postprocessing")
                # We do not call Parent method as there is no need to re-compute offline operators
            else:
                # Call Parent, which may eventually raise an error
                EllipticCoerciveReducedProblem_DerivedClass.build_reduced_operators(self, current_stage)
            
        def assemble_operator(self, term, current_stage="online"):
            if term == "projection_truth_snapshots":
                assert current_stage in ("online", "offline_rectification_postprocessing")
                if current_stage == "online": # load from file
                    self.operator["projection_truth_snapshots"].load(self.folder["reduced_operators"], "projection_truth_snapshots")
                    return self.operator["projection_truth_snapshots"]
                elif current_stage == "offline_rectification_postprocessing":
                    assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                    inner_product = self.truth_problem.inner_product[0]
                    for n in range(1, self.N + 1):
                        assert len(self.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                        inner_product_n = self.inner_product[:n, :n][0]
                        basis_functions_n = self.basis_functions[:n]
                        projection_truth_snapshots_expansion = OnlineAffineExpansionStorage(1)
                        projection_truth_snapshots = OnlineMatrix(n, n)
                        for (i, snapshot_i) in enumerate(self.snapshots[:n]):
                            projected_truth_snapshot_i = OnlineFunction(n)
                            solver = LinearSolver(inner_product_n, projected_truth_snapshot_i, transpose(basis_functions_n)*inner_product*snapshot_i)
                            solver.set_parameters(self._linear_solver_parameters)
                            solver.solve()
                            for j in range(n):
                                projection_truth_snapshots[j, i] = projected_truth_snapshot_i.vector()[j]
                        projection_truth_snapshots_expansion[0] = projection_truth_snapshots
                        print("\tcondition number for n = " + str(n) + ": " + str(cond(projection_truth_snapshots)))
                        self.operator["projection_truth_snapshots"][:n, :n] = projection_truth_snapshots_expansion
                    # Save
                    self.operator["projection_truth_snapshots"].save(self.folder["reduced_operators"], "projection_truth_snapshots")
                    return self.operator["projection_truth_snapshots"]
                else:
                    raise ValueError("Invalid stage in assemble_operator().")
            elif term == "projection_reduced_snapshots":
                assert current_stage in ("online", "offline_rectification_postprocessing")
                if current_stage == "online": # load from file
                    self.operator["projection_reduced_snapshots"].load(self.folder["reduced_operators"], "projection_reduced_snapshots")
                    return self.operator["projection_reduced_snapshots"]
                elif current_stage == "offline_rectification_postprocessing":
                    # Backup mu
                    bak_mu = self.mu
                    # Prepare rectification for all possible online solve arguments
                    for n in range(1, self.N + 1):
                        print("\tcondition number for n = " + str(n))
                        projection_reduced_snapshots_expansion = OnlineAffineExpansionStorage(len(self.online_solve_kwargs_without_rectification))
                        for (q, online_solve_kwargs) in enumerate(self.online_solve_kwargs_without_rectification):
                            projection_reduced_snapshots = OnlineMatrix(n, n)
                            for (i, mu_i) in enumerate(self.snapshots_mu[:n]):
                                self.set_mu(mu_i)
                                projected_reduced_snapshot_i = self.solve(n, **online_solve_kwargs)
                                for j in range(n):
                                    projection_reduced_snapshots[j, i] = projected_reduced_snapshot_i.vector()[j]
                            projection_reduced_snapshots_expansion[q] = projection_reduced_snapshots
                            print("\t\tonline solve options " + str(dict(self.online_solve_kwargs_with_rectification[q])) + ": " + str(cond(projection_reduced_snapshots)))
                        self.operator["projection_reduced_snapshots"][:n, :n] = projection_reduced_snapshots_expansion
                    # Save and restore previous mu
                    self.set_mu(bak_mu)
                    self.operator["projection_reduced_snapshots"].save(self.folder["reduced_operators"], "projection_reduced_snapshots")
                    return self.operator["projection_reduced_snapshots"]
                else:
                    raise ValueError("Invalid stage in assemble_operator().")
            else:
                return EllipticCoerciveReducedProblem_DerivedClass.assemble_operator(self, term, current_stage)
            
        def _solve(self, N, **kwargs):
            EllipticCoerciveReducedProblem_DerivedClass._solve(self, N, **kwargs)
            if kwargs["online_rectification"]:
                q = self.online_solve_kwargs_with_rectification.index(kwargs)
                intermediate_solution = OnlineFunction(N)
                solver = LinearSolver(self.operator["projection_reduced_snapshots"][:N, :N][q], intermediate_solution, self._solution.vector())
                solver.set_parameters(self._linear_solver_parameters)
                solver.solve()
                self._solution.vector()[:] = self.operator["projection_truth_snapshots"][:N, :N][0]*intermediate_solution
                
        def estimate_error(self):
            if self._disable_error_estimation:
                return NotImplemented
            else:
                return EllipticCoerciveReducedProblem_DerivedClass.estimate_error(self)
                
    # return value (a class) for the decorator
    return OnlineRectificationDecoratedReducedProblem_Class
