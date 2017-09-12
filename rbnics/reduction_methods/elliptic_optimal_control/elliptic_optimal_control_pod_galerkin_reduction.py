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


from rbnics.backends import FunctionsList
from rbnics.utils.decorators import ReductionMethodFor
from rbnics.problems.elliptic_optimal_control.elliptic_optimal_control_problem import EllipticOptimalControlProblem
from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearPODGalerkinReduction
from rbnics.reduction_methods.elliptic_optimal_control.elliptic_optimal_control_reduction_method import EllipticOptimalControlReductionMethod

EllipticOptimalControlPODGalerkinReduction_Base = LinearPODGalerkinReduction(EllipticOptimalControlReductionMethod(DifferentialProblemReductionMethod))

# Base class containing the interface of a POD-Galerkin ROM
# for elliptic coercive problems
@ReductionMethodFor(EllipticOptimalControlProblem, "PODGalerkin")
class EllipticOptimalControlPODGalerkinReduction(EllipticOptimalControlPODGalerkinReduction_Base):
    
    ## Compute basis functions performing POD: overridden to handle aggregated spaces
    def compute_basis_functions(self):
        # Carry out POD
        Z = dict()
        N = dict()
        for component in self.truth_problem.components:
            print("# POD for component", component)
            POD = self.POD[component]
            assert self.tol[component] == 0. # TODO first negelect tolerances, then compute the max of N for each aggregated pair
            (_, Z[component], N[component]) = POD.apply(self.Nmax, self.tol[component])
            POD.print_eigenvalues(N[component])
            POD.save_eigenvalues_file(self.folder["post_processing"], "eigs_" + component)
            POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy_" + component)
        
        # Store POD modes related to control as usual
        self.reduced_problem.Z.enrich(Z["u"], component="u")
        self.reduced_problem.N["u"] += N["u"]
        
        # Aggregate POD modes related to state and adjoint
        for component_to in ("y", "p"):
            for i in range(self.Nmax): # TODO should have been N[component_from], but cannot switch the next line
                for component_from in ("y", "p"):
                    self.reduced_problem.Z.enrich(Z[component_from][i], component={component_from: component_to})
                self.reduced_problem.N[component_to] += 2
        
        # Save
        self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
    
