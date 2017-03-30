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
## @file parabolic_coercive_reduced_problem.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.problems.elliptic_coercive import EllipticCoercivePODGalerkinReducedProblem
from rbnics.problems.parabolic_coercive.parabolic_coercive_reduced_problem import ParabolicCoerciveReducedProblem
from rbnics.utils.decorators import Extends, override, ReducedProblemFor
from rbnics.problems.base import TimeDependentPODGalerkinReducedProblem
from rbnics.problems.parabolic_coercive.parabolic_coercive_problem import ParabolicCoerciveProblem
from rbnics.reduction_methods.parabolic_coercive import ParabolicCoercivePODGalerkinReduction

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParabolicCoerciveReducedOrderModelBase
#

ParabolicCoercivePODGalerkinReducedProblem_Base = ParabolicCoerciveReducedProblem(TimeDependentPODGalerkinReducedProblem(EllipticCoercivePODGalerkinReducedProblem))

# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(ParabolicCoercivePODGalerkinReducedProblem_Base) # needs to be first in order to override for last the methods
@ReducedProblemFor(ParabolicCoerciveProblem, ParabolicCoercivePODGalerkinReduction)
class ParabolicCoercivePODGalerkinReducedProblem(ParabolicCoercivePODGalerkinReducedProblem_Base):
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        ParabolicCoercivePODGalerkinReducedProblem_Base.__init__(self, truth_problem, **kwargs)
        
