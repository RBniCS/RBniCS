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


from rbnics.utils.decorators import ReductionMethodFor
from rbnics.problems.elliptic_optimal_control.elliptic_optimal_control_problem import EllipticOptimalControlProblem
from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearRBReduction
from rbnics.reduction_methods.elliptic_optimal_control.elliptic_optimal_control_reduction_method import EllipticOptimalControlReductionMethod

EllipticOptimalControlRBReduction_Base = LinearRBReduction(EllipticOptimalControlReductionMethod(DifferentialProblemReductionMethod))

@ReductionMethodFor(EllipticOptimalControlProblem, "ReducedBasis")
class EllipticOptimalControlRBReduction(EllipticOptimalControlRBReduction_Base):
    def update_basis_matrix(self, snapshot):
        # Aggregate snapshots components related to state and adjoint
        for component_to in ("y", "p"):
            for component_from in ("y", "p"):
                new_basis_function = self.GS[component_to].apply(snapshot, self.reduced_problem.basis_functions[component_to][self.reduced_problem.N_bc[component_to]:], component={component_from: component_to})
                self.reduced_problem.basis_functions.enrich(new_basis_function, component=component_to)
                self.reduced_problem.N[component_to] += 1
                
        # Store snapshots components related to control as usual
        new_basis_function = self.GS["u"].apply(snapshot, self.reduced_problem.basis_functions["u"][self.reduced_problem.N_bc["u"]:])
        self.reduced_problem.basis_functions.enrich(new_basis_function, component="u")
        self.reduced_problem.N["u"] += 1
        
        # Save
        self.reduced_problem.basis_functions.save(self.reduced_problem.folder["basis"], "basis")
