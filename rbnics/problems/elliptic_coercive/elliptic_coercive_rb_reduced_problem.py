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

from math import sqrt
from numpy import isclose
from rbnics.backends import product, sum, transpose
from rbnics.problems.base import LinearRBReducedProblem, ParametrizedReducedDifferentialProblem, PrimalDualReducedProblem
from rbnics.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from rbnics.problems.elliptic_coercive.elliptic_coercive_reduced_problem import EllipticCoerciveReducedProblem
from rbnics.reduction_methods.elliptic_coercive import EllipticCoerciveRBReduction
from rbnics.utils.decorators import ReducedProblemFor

EllipticCoerciveRBReducedProblem_Base = LinearRBReducedProblem(EllipticCoerciveReducedProblem(ParametrizedReducedDifferentialProblem))

# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
# The following implementation will be retained if no output is provided in the "s" term
@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveRBReduction)
class EllipticCoerciveRBReducedProblem(EllipticCoerciveRBReducedProblem_Base):
    
    # Default initialization of members.
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        EllipticCoerciveRBReducedProblem_Base.__init__(self, truth_problem, **kwargs)
        
        # Skip useless Riesz products
        self.riesz_terms = ["f", "a"]
        self.riesz_product_terms = [("f", "f"), ("a", "f"), ("a", "a")]
    
    # Return an error bound for the current solution
    def estimate_error(self):
        eps2 = self.get_residual_norm_squared()
        alpha = self.get_stability_factor()
        assert eps2 >= 0. or isclose(eps2, 0.)
        assert alpha >= 0.
        return sqrt(abs(eps2))/alpha
        
    # Return a relative error bound for the current solution
    def estimate_relative_error(self):
        return NotImplemented
    
    # Return an error bound for the current output
    def estimate_error_output(self):
        return NotImplemented
        
    # Return a relative error bound for the current output
    def estimate_relative_error_output(self):
        return NotImplemented
        
    # Return the numerator of the error bound for the current solution
    def get_residual_norm_squared(self):
        N = self._solution.N
        theta_a = self.compute_theta("a")
        theta_f = self.compute_theta("f")
        return (
              sum(product(theta_f, self.riesz_product["f", "f"], theta_f))
            + 2.0*(transpose(self._solution)*sum(product(theta_a, self.riesz_product["a", "f"][:N], theta_f)))
            + transpose(self._solution)*sum(product(theta_a, self.riesz_product["a", "a"][:N, :N], theta_a))*self._solution
        )
    
# Add dual reduced problem if an output is provided in the term "s"
def _problem_has_output(truth_problem, reduction_method, **kwargs):
    try:
        truth_problem.compute_theta("s")
    except ValueError:
        return False
    else:
        return True
        
EllipticCoerciveRBReducedProblem_PrimalDual_Base = PrimalDualReducedProblem(EllipticCoerciveRBReducedProblem)

@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveRBReduction, replaces=EllipticCoerciveRBReducedProblem, replaces_if=_problem_has_output)
class EllipticCoerciveRBReducedProblem_PrimalDual(EllipticCoerciveRBReducedProblem_PrimalDual_Base):
    pass
