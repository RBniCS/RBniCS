# Copyright (C) 2015-2017 by the RBniCS authors
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

from rbnics.backends import BasisFunctionsMatrix, transpose
from rbnics.backends.online import OnlineEigenSolver
from rbnics.utils.decorators import PreserveClassName, ReductionMethodDecoratorFor
from rbnics.utils.io import ExportableList
from backends.dolfin import NonHierarchicalBasisFunctionsMatrix
from backends.online import OnlineSolveKwargsGenerator
from problems import OnlineVanishingViscosity

@ReductionMethodDecoratorFor(OnlineVanishingViscosity)
def OnlineVanishingViscosityDecoratedReductionMethod(EllipticCoerciveReductionMethod_DerivedClass):
    
    @PreserveClassName
    class OnlineVanishingViscosityDecoratedReductionMethod_Class(EllipticCoerciveReductionMethod_DerivedClass):
        
        def _offline(self):
            # Change default online solve arguments during offline stage to use online stabilization
            # instead of vanishing viscosity one (which will be prepared in a postprocessing stage)
            self.reduced_problem._online_solve_default_kwargs["online_stabilization"] = True
            self.reduced_problem._online_solve_default_kwargs["online_vanishing_viscosity"] = False
            self.reduced_problem.OnlineSolveKwargs = OnlineSolveKwargsGenerator(**self.reduced_problem._online_solve_default_kwargs)
            
            # Call standard offline phase
            EllipticCoerciveReductionMethod_DerivedClass._offline(self)
            
            print("==============================================================")
            print("=" + "{:^60}".format(self.label + " offline vanishing viscosity postprocessing phase begins") + "=")
            print("==============================================================")
            print("")
            
            # Prepare storage for copy of lifting basis functions matrix
            lifting_Z = BasisFunctionsMatrix(self.truth_problem.V)
            lifting_Z.init(self.truth_problem.components)
            # Copy current lifting basis functions to lifting_Z
            N_bc = self.reduced_problem.N_bc
            for i in range(N_bc):
                lifting_Z.enrich(self.reduced_problem.Z[i])
            # Prepare storage for unrotated basis functions matrix, without lifting
            unrotated_Z = BasisFunctionsMatrix(self.truth_problem.V)
            unrotated_Z.init(self.truth_problem.components)
            # Copy current basis functions (except lifting) to unrotated_Z
            N = self.reduced_problem.N
            for i in range(N_bc, N):
                unrotated_Z.enrich(self.reduced_problem.Z[i])
                
            # Prepare new storage for non-hierarchical basis functions matrix
            Z = NonHierarchicalBasisFunctionsMatrix(self.truth_problem.V)
            Z.init(self.truth_problem.components)
            # Rotated basis functions matrix are not hierarchical, i.e. a different
            # rotation will be applied for each basis size n.
            for n in range(1, N + 1):
                # Prepare storage for rotated basis functions matrix
                rotated_Z = BasisFunctionsMatrix(self.truth_problem.V)
                rotated_Z.init(self.truth_problem.components)
                # Rotate basis
                print("rotate basis functions matrix for n =", n)
                truth_operator_k = self.truth_problem.operator["k"]
                truth_operator_m = self.truth_problem.operator["m"]
                assert len(truth_operator_k) == 1
                assert len(truth_operator_m) == 1
                reduced_operator_k = transpose(unrotated_Z[:n])*truth_operator_k[0]*unrotated_Z[:n]
                reduced_operator_m = transpose(unrotated_Z[:n])*truth_operator_m[0]*unrotated_Z[:n]
                rotation_eigensolver = OnlineEigenSolver(unrotated_Z[:n], reduced_operator_k, reduced_operator_m)
                parameters = {
                    "problem_type": "hermitian",
                    "spectrum": "smallest real"
                }
                rotation_eigensolver.set_parameters(parameters)
                rotation_eigensolver.solve()
                # Store and save rotated basis
                rotation_eigenvalues = ExportableList("text")
                rotation_eigenvalues.extend([rotation_eigensolver.get_eigenvalue(i)[0] for i in range(n)])
                for i in range(0, n):
                    print("lambda_" + str(i) + " = " + str(rotation_eigenvalues[i]))
                rotation_eigenvalues.save(self.folder["post_processing"], "rotation_eigs_n=" + str(n))
                for i in range(N_bc):
                    rotated_Z.enrich(lifting_Z[i])
                for i in range(0, n):
                    (eigenvector_i, _) = rotation_eigensolver.get_eigenvector(i)
                    rotated_Z.enrich(unrotated_Z[:n]*eigenvector_i)
                Z[:n] = rotated_Z
                # Attach eigenvalues to the vanishing viscosity reduced operator
                self.reduced_problem.vanishing_viscosity_eigenvalues.append(rotation_eigenvalues)
                
            # Save Z and attach it to reduced problem
            Z.save(self.reduced_problem.folder["basis"], "basis")
            self.reduced_problem.Z = Z
            
            # Re-compute all reduced operators, since the basis functions have changed
            print("build reduced operators")
            self.reduced_problem.build_reduced_operators()
            # Re-compute all error estimation operators, since the basis functions have changed
            if hasattr(self.reduced_problem, "build_error_estimation_operators"):
                for term in self.reduced_problem.riesz_terms:
                    if self.reduced_problem.terms_order[term] > 1:
                        for q in range(self.reduced_problem.Q[term]):
                            self.reduced_problem.riesz[term][q].clear()
                print("build operators for error estimation")
                self.reduced_problem.build_error_estimation_operators()
            
            print("==============================================================")
            print("=" + "{:^60}".format(self.label + " offline vanishing viscosity postprocessing phase ends") + "=")
            print("==============================================================")
            print("")
            
            # Restore default online solve arguments for online stage
            self.reduced_problem._online_solve_default_kwargs["online_stabilization"] = False
            self.reduced_problem._online_solve_default_kwargs["online_vanishing_viscosity"] = True
            self.reduced_problem.OnlineSolveKwargs = OnlineSolveKwargsGenerator(**self.reduced_problem._online_solve_default_kwargs)
            
        def update_basis_matrix(self, snapshot): # same as Parent, except a different filename is used when saving
            assert len(self.truth_problem.components) is 1
            self.reduced_problem.Z.enrich(snapshot)
            self.GS.apply(self.reduced_problem.Z, self.reduced_problem.N_bc)
            self.reduced_problem.N += 1
            self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "unrotated_basis")
        
    # return value (a class) for the decorator
    return OnlineVanishingViscosityDecoratedReductionMethod_Class
