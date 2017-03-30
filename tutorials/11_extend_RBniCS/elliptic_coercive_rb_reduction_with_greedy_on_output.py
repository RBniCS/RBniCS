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
## @file elliptic_coercive_rb.py
#  @brief Implementation of the reduced basis method for (compliant) elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.utils.decorators import Extends, override, ReductionMethodFor
from rbnics.problems.elliptic_coercive import EllipticCoerciveProblem
from rbnics.reduction_methods.elliptic_coercive import EllipticCoerciveRBReduction

def _has_use_dual_flag(truth_problem, **kwargs):
    return hasattr(truth_problem, "use_dual") # TODO Do not attach it to truth problem, rather use kwargs

@Extends(EllipticCoerciveRBReduction) # needs to be first in order to override for last the methods
@ReductionMethodFor(EllipticCoerciveProblem, "ReducedBasis", replaces=EllipticCoerciveRBReduction, replaces_if=_has_use_dual_flag)
class EllipticCoerciveRBReductionWithGreedyOnOutput(EllipticCoerciveRBReduction):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem):
        # Call the parent initialization
        EllipticCoerciveRBReduction.__init__(self, truth_problem)
        
        raise NotImplementedError("Tutorial 11 not implemented yet.") # TODO
                
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
        
