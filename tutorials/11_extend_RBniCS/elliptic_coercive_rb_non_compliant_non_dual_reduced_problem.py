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

from rbnics.problems.elliptic_coercive import EllipticCoerciveProblem, EllipticCoerciveRBReducedProblem, EllipticCoerciveRBNonCompliantReducedProblem
from rbnics.utils.decorators import Extends, override, ReducedProblemFor
from elliptic_coercive_rb_reduction_with_greedy_on_output import EllipticCoerciveRBReductionWithGreedyOnOutput

def _do_not_use_dual(truth_problem, **kwargs):
    return not truth_problem.use_dual # TODO Do not attach it to truth problem, rather use kwargs

@Extends(EllipticCoerciveRBReducedProblem) # needs to be first in order to override for last the methods
@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveRBReductionWithGreedyOnOutput, replaces=EllipticCoerciveRBNonCompliantReducedProblem, replaces_if=_do_not_use_dual)
class EllipticCoerciveRBNonCompliantNonDualReducedProblem(EllipticCoerciveRBReducedProblem):
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem):
        # Call to parent
        EllipticCoerciveRBReducedProblem.__init__(self, truth_problem)
        
        raise NotImplementedError("Tutorial 11 not implemented yet.") # TODO

