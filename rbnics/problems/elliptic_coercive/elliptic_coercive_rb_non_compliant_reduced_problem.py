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

from rbnics.utils.decorators import ReducedProblemFor
from rbnics.backends import product, sum, transpose
from rbnics.problems.base import PrimalDualReducedProblem
from rbnics.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from rbnics.problems.elliptic_coercive.elliptic_coercive_rb_reduced_problem import EllipticCoerciveRBReducedProblem
from rbnics.reduction_methods.elliptic_coercive import EllipticCoerciveRBReduction
from rbnics.reduction_methods.elliptic_coercive.elliptic_coercive_rb_non_compliant_reduction import _problem_is_noncompliant

EllipticCoerciveRBNonCompliantReducedProblem_Base = PrimalDualReducedProblem(EllipticCoerciveRBReducedProblem)

@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveRBReduction, replaces=EllipticCoerciveRBReducedProblem, replaces_if=_problem_is_noncompliant)
class EllipticCoerciveRBNonCompliantReducedProblem(EllipticCoerciveRBNonCompliantReducedProblem_Base):
    
    # Perform an online evaluation of the non compliant output
    def _compute_output(self, N):
        self._output = transpose(self._solution)*sum(product(self.compute_theta("s"), self.operator["s"][:N]))
            
