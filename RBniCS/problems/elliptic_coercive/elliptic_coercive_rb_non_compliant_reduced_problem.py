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

from RBniCS.problems.elliptic_coercive.elliptic_coercive_rb_reduced_problem import EllipticCoerciveRBReducedProblem
from RBniCS.utils.decorators import Extends, override, ReducedProblemFor
from RBniCS.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from RBniCS.reduction_methods.elliptic_coercive import EllipticCoerciveRBReduction

def _problem_is_noncompliant(truth_problem):
    try:
        theta_s = truth_problem.compute_theta("s")
    except ValueError:
        return False
    else:
        return True

@Extends(EllipticCoerciveRBReducedProblem) # needs to be first in order to override for last the methods
@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveRBReduction, replaces=EllipticCoerciveRBReducedProblem, replaces_if=_problem_is_noncompliant)
class EllipticCoerciveRBNonCompliantReducedProblem(EllipticCoerciveRBReducedProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem):
        # Call to parent
        EllipticCoerciveRBReducedProblem.__init__(self, truth_problem)
        
        #raise NotImplementedError("This class has not been implemented yet.") # TODO

        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
