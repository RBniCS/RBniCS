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


from rbnics.backends import GramSchmidt
from rbnics.utils.decorators import Extends, override, ReductionMethodFor
from rbnics.problems.elliptic_optimal_control.elliptic_optimal_control_problem import EllipticOptimalControlProblem
from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearRBReduction
from rbnics.reduction_methods.elliptic_optimal_control.elliptic_optimal_control_reduction_method import EllipticOptimalControlReductionMethod

EllipticOptimalControlRBReduction_Base = LinearRBReduction(EllipticOptimalControlReductionMethod(DifferentialProblemReductionMethod))

# Base class containing the interface of a RB ROM
# for elliptic coercive problems
@Extends(EllipticOptimalControlRBReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(EllipticOptimalControlProblem, "ReducedBasis")
class EllipticOptimalControlRBReduction(EllipticOptimalControlRBReduction_Base):
    def update_basis_matrix(self, snapshot):
        # Aggregate snapshots components related to state and adjoint
        for component_to in ("y", "p"):
            for component_from in ("y", "p"):
                self.reduced_problem.Z.enrich(snapshot, component={component_from: component_to})
                self.GS[component_to].apply(self.reduced_problem.Z[component_to], self.reduced_problem.N_bc[component_to])
                self.reduced_problem.N[component_to] += 1
                
        # Store snapshots components related to control as usual
        self.reduced_problem.Z.enrich(snapshot, component="u")
        self.GS["u"].apply(self.reduced_problem.Z["u"], self.reduced_problem.N_bc["u"])
        self.reduced_problem.N["u"] += 1
        
        # Save
        self.reduced_problem.Z.save(self.reduced_problem.folder["basis"], "basis")
