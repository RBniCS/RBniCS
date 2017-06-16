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

import types
from rbnics.backends import product, sum, transpose
from rbnics.utils.decorators import Extends, ReductionMethodFor
from rbnics.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from rbnics.problems.elliptic_coercive.elliptic_coercive_problem_dual import EllipticCoerciveProblem_Dual
from rbnics.reduction_methods.base import PrimalDualReductionMethod
from rbnics.reduction_methods.elliptic_coercive.elliptic_coercive_rb_reduction import EllipticCoerciveRBReduction

def _problem_is_noncompliant(truth_problem, **kwargs):
    try:
        theta_s = truth_problem.compute_theta("s")
    except ValueError:
        return False
    else:
        # Make sure to add "s" to available terms
        assert (
            ("s" in truth_problem.terms)
                ==
            ("s" in truth_problem.terms_order)
        )
        if not "s" in truth_problem.terms:
            truth_problem.terms.append("s")
            truth_problem.terms_order["s"] = 1
            # Change the computation of the output to use "s"
            def _compute_output(self_):
                assembled_output_operator = sum(product(self_.compute_theta("s"), self_.operator["s"]))
                self_._output = transpose(assembled_output_operator)*self_._solution
                return self_._output
            truth_problem._compute_output = types.MethodType(_compute_output, truth_problem)
        #
        return True

EllipticCoerciveRBNonCompliantReduction_Base = PrimalDualReductionMethod(EllipticCoerciveProblem_Dual)(EllipticCoerciveRBReduction)

# Base class containing the interface of the RB method
# for non compliant elliptic coercive problems
@Extends(EllipticCoerciveRBNonCompliantReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(EllipticCoerciveProblem, "ReducedBasis", replaces=EllipticCoerciveRBReduction, replaces_if=_problem_is_noncompliant)
class EllipticCoerciveRBNonCompliantReduction(EllipticCoerciveRBNonCompliantReduction_Base):
    pass
        
