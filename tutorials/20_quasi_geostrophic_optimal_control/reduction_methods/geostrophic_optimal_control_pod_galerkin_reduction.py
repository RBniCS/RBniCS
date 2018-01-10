# Copyright (C) 2015-2018 by the RBniCS authors
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

from rbnics.utils.decorators import ReductionMethodFor
from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearPODGalerkinReduction
from problems import GeostrophicOptimalControlProblem
from .geostrophic_optimal_control_reduction_method import GeostrophicOptimalControlReductionMethod

GeostrophicOptimalControlPODGalerkinReduction_Base = LinearPODGalerkinReduction(GeostrophicOptimalControlReductionMethod(DifferentialProblemReductionMethod))

@ReductionMethodFor(GeostrophicOptimalControlProblem, "PODGalerkin")
class GeostrophicOptimalControlPODGalerkinReduction(GeostrophicOptimalControlPODGalerkinReduction_Base):
    
    # Compute basis functions performing POD: overridden to handle aggregated spaces
    def compute_basis_functions(self):
        # Carry out POD
        basis_functions = dict()
        N = dict()
        for component in self.truth_problem.components:
            print("# POD for component", component)
            POD = self.POD[component]
            assert self.tol[component] == 0. # TODO first negelect tolerances, then compute the max of N for each aggregated pair
            (_, basis_functions[component], N[component]) = POD.apply(self.Nmax, self.tol[component])
            POD.print_eigenvalues(N[component])
            POD.save_eigenvalues_file(self.folder["post_processing"], "eigs_" + component)
            POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy_" + component)
        
        # Store POD modes related to control as usual
        self.reduced_problem.basis_functions.enrich(basis_functions["u"], component="u")
        self.reduced_problem.N["u"] += N["u"]
        
        # Aggregate POD modes related to state and adjoint
        for pair in (("ypsi", "ppsi"), ("yq", "pq")):
            for component_to in pair:
                for i in range(self.Nmax): # TODO should have been N[component_from], but cannot switch the next line
                    for component_from in pair:
                        self.reduced_problem.basis_functions.enrich(basis_functions[component_from][i], component={component_from: component_to})
                    self.reduced_problem.N[component_to] += 2
        
        # Save
        self.reduced_problem.basis_functions.save(self.reduced_problem.folder["basis"], "basis")
