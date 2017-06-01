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

from math import sqrt
from rbnics.utils.decorators import Extends, override, ReductionMethodFor
from rbnics.problems.parabolic_coercive.parabolic_coercive_problem import ParabolicCoerciveProblem
from rbnics.reduction_methods.base import TimeDependentRBReduction
from rbnics.reduction_methods.elliptic_coercive import EllipticCoerciveRBReduction
from rbnics.reduction_methods.parabolic_coercive.parabolic_coercive_reduction_method import ParabolicCoerciveReductionMethod

ParabolicCoerciveRBReduction_Base = ParabolicCoerciveReductionMethod(TimeDependentRBReduction(EllipticCoerciveRBReduction))

# Base class containing the interface of a RB ROM
# for parabolic coercive problems
@Extends(ParabolicCoerciveRBReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(ParabolicCoerciveProblem, "ReducedBasis")
class ParabolicCoerciveRBReduction(ParabolicCoerciveRBReduction_Base):
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call the parent initialization
        ParabolicCoerciveRBReduction_Base.__init__(self, truth_problem, **kwargs)
        
