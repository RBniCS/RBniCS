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
from problems import OnlineVanishingViscosity

@ReductionMethodDecoratorFor(OnlineVanishingViscosity)
def OnlineVanishingViscosityDecoratedReductionMethod(EllipticCoerciveReductionMethod_DerivedClass):
    
    @PreserveClassName
    class OnlineVanishingViscosityDecoratedReductionMethod_Class(EllipticCoerciveReductionMethod_DerivedClass):
        
        def _offline(self):
            # Call standard offline phase
            EllipticCoerciveReductionMethod_DerivedClass._offline(self)
            # Backup computed basis functions
            Z = self.reduced_problem.Z
            Z.save(self.reduced_problem.folder["basis"], "unrotated_basis")
            
            print("==============================================================")
            print("=" + "{:^60}".format(self.label + " offline vanishing viscosity postprocessing phase begins") + "=")
            print("==============================================================")
            print("")
            
            # Prepare storage for rotated basis functions matrix
            rotated_Z = BasisFunctionsMatrix(self.truth_problem.V)
            rotated_Z.init(self.truth_problem.components)
            # Prepare storage for unrotated basis functions matrix, without lifting
            unrotated_Z = BasisFunctionsMatrix(self.truth_problem.V)
            unrotated_Z.init(self.truth_problem.components)
            # Copy current basis functions (except lifting) to unrotated_Z
            N_bc = self.reduced_problem.N_bc
            N = self.reduced_problem.N
            for i in range(N_bc, N):
                unrotated_Z.enrich(Z[i])
            # Rotate basis
            print("rotate basis functions matrix")
            truth_operator_k = self.truth_problem.operator["k"]
            truth_operator_m = self.truth_problem.operator["m"]
            assert len(truth_operator_k) == 1
            assert len(truth_operator_m) == 1
            reduced_operator_k = transpose(unrotated_Z)*truth_operator_k[0]*unrotated_Z
            reduced_operator_m = transpose(unrotated_Z)*truth_operator_m[0]*unrotated_Z
            rotation_eigensolver = OnlineEigenSolver(unrotated_Z, reduced_operator_k, reduced_operator_m)
            parameters = {
                "problem_type": "hermitian",
                "spectrum": "smallest real"
            }
            rotation_eigensolver.set_parameters(parameters)
            rotation_eigensolver.solve()
            # Store and save rotated basis
            rotation_eigenvalues = ExportableList("text")
            rotation_eigenvalues.extend([rotation_eigensolver.get_eigenvalue(i)[0] for i in range(N)])
            for i in range(N_bc, N):
                print("lambda_" + str(i) + " = " + str(rotation_eigenvalues[i]))
            rotation_eigenvalues.save(self.folder["post_processing"], "rotation_eigs")
            for i in range(N_bc):
                rotated_Z.enrich(Z[i])
            for i in range(N_bc, N):
                (eigenvector_i, _) = rotation_eigensolver.get_eigenvector(i - N_bc)
                rotated_Z.enrich(unrotated_Z*eigenvector_i)
            Z.clear()
            for i in range(N):
                Z.enrich(rotated_Z[i])
            Z.save(self.reduced_problem.folder["basis"], "basis")
            # Attach eigenvalues to the vanishing viscosity reduced operator
            self.reduced_problem.vanishing_viscosity_eigenvalues = rotation_eigenvalues
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
        
    # return value (a class) for the decorator
    return OnlineVanishingViscosityDecoratedReductionMethod_Class
