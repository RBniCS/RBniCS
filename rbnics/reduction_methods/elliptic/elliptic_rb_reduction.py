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

from rbnics.problems.elliptic.elliptic_problem import EllipticProblem
from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearRBReduction
from rbnics.reduction_methods.elliptic.elliptic_reduction_method import EllipticReductionMethod
from rbnics.utils.decorators import ReductionMethodFor

EllipticRBReduction_Base = LinearRBReduction(EllipticReductionMethod(DifferentialProblemReductionMethod))

# Base class containing the interface of the RB method
# for elliptic problems
@ReductionMethodFor(EllipticProblem, "ReducedBasis")
class EllipticRBReduction(EllipticRBReduction_Base):
    pass
