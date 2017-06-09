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

from rbnics.utils.decorators import Extends, override, PrimalDualReducedProblem, ReducedProblemFor
from rbnics.backends import product, sum, transpose
from rbnics.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from rbnics.problems.elliptic_coercive.elliptic_coercive_rb_reduced_problem import EllipticCoerciveRBReducedProblem
from rbnics.reduction_methods.elliptic_coercive import EllipticCoerciveRBReduction
from rbnics.reduction_methods.elliptic_coercive.elliptic_coercive_rb_non_compliant_reduction import _problem_is_noncompliant

EllipticCoerciveRBNonCompliantReducedProblem_Base = PrimalDualReducedProblem(EllipticCoerciveRBReducedProblem)

@Extends(EllipticCoerciveRBNonCompliantReducedProblem_Base) # needs to be first in order to override for last the methods
@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveRBReduction, replaces=EllipticCoerciveRBReducedProblem, replaces_if=_problem_is_noncompliant)
class EllipticCoerciveRBNonCompliantReducedProblem(EllipticCoerciveRBNonCompliantReducedProblem_Base):
    
    # Perform an online evaluation of the non compliant output
    @override
    def _compute_output(self, N):
        self._output = transpose(self._solution)*sum(product(self.compute_theta("s"), self.operator["s"][:N]))
        
    ## Compute the Riesz representation of term
    @override
    def compute_riesz(self, term):
        if term == "s":
            pass
        else:
            return EllipticCoerciveRBNonCompliantReducedProblem_Base.compute_riesz(self, term)
            
    ## Assemble operators for error estimation
    @override
    def assemble_error_estimation_operators(self, term, current_stage="online"):
        if term in (("a", "s"), ("f", "s"), ("s", "s")):
            pass
        else:
            return EllipticCoerciveRBNonCompliantReducedProblem_Base.assemble_error_estimation_operators(self, term, current_stage)
            
