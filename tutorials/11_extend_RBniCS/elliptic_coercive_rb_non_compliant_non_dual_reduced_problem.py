# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file elliptic_coercive_reduced_problem.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.elliptic_coercive import EllipticCoerciveProblem, EllipticCoerciveRBReducedProblem, EllipticCoerciveRBNonCompliantReducedProblem
from RBniCS.utils.decorators import Extends, override, ReducedProblemFor
from elliptic_coercive_rb_reduction_with_greedy_on_output import EllipticCoerciveRBReductionWithGreedyOnOutput

def _do_not_use_dual(truth_problem):
    return not truth_problem.use_dual

@Extends(EllipticCoerciveRBReducedProblem) # needs to be first in order to override for last the methods
@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveRBReductionWithGreedyOnOutput, replaces=EllipticCoerciveRBNonCompliantReducedProblem, replaces_if=_do_not_use_dual)
class EllipticCoerciveRBNonCompliantNonDualReducedProblem(EllipticCoerciveRBReducedProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem):
        # Call to parent
        EllipticCoerciveRBReducedProblem.__init__(self, truth_problem)
        
        raise NotImplementedError("Tutorial 11 not implemented yet.") # TODO

        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
