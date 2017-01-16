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
## @file elliptic_coercive_reduced_problem.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators import Extends, override, PrimalDualReducedProblem, ReducedProblemFor
from RBniCS.backends import product, sum, transpose
from RBniCS.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from RBniCS.problems.elliptic_coercive.elliptic_coercive_rb_reduced_problem import EllipticCoerciveRBReducedProblem
from RBniCS.reduction_methods.elliptic_coercive import EllipticCoerciveRBReduction
from RBniCS.reduction_methods.elliptic_coercive.elliptic_coercive_rb_non_compliant_reduction import _problem_is_noncompliant

@Extends(EllipticCoerciveRBReducedProblem) # needs to be first in order to override for last the methods
@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveRBReduction, replaces=EllipticCoerciveRBReducedProblem, replaces_if=_problem_is_noncompliant)
@PrimalDualReducedProblem
class EllipticCoerciveRBNonCompliantReducedProblem(EllipticCoerciveRBReducedProblem):
    
    # Perform an online evaluation of the non compliant output
    @override
    def output(self):
        N = self._solution.N
        assembled_output_operator = sum(product(self.compute_theta("s"), self.operator["s"][:N]))
        self._output = transpose(assembled_output_operator)*self._solution
        return self._output
        
    ## Compute the Riesz representation of term
    @override
    def compute_riesz(self, term):
        if term == "s":
            pass
        else:
            return EllipticCoerciveRBReducedProblem.compute_riesz(self, term)
            
    ## Assemble operators for error estimation
    @override
    def assemble_error_estimation_operators(self, term, current_stage="online"):
        if term in ("riesz_product_as", "riesz_product_fs", "riesz_product_ss"):
            pass
        else:
            return EllipticCoerciveRBReducedProblem.assemble_error_estimation_operators(self, term, current_stage)
            
