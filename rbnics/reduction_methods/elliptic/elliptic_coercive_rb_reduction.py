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

from rbnics.problems.elliptic.elliptic_coercive_problem import EllipticCoerciveProblem
from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearRBReduction
from rbnics.reduction_methods.elliptic.elliptic_coercive_reduction_method import EllipticCoerciveReductionMethod
from rbnics.utils.decorators import ReductionMethodFor

EllipticCoerciveRBReduction_Base = LinearRBReduction(EllipticCoerciveReductionMethod(DifferentialProblemReductionMethod))

# Base class containing the interface of the RB method
# for (compliant) elliptic coercive problems
# The following implementation will be retained if no output is provided in the "s" term
@ReductionMethodFor(EllipticCoerciveProblem, "ReducedBasis")
class EllipticCoerciveRBReduction(EllipticCoerciveRBReduction_Base):
    pass
