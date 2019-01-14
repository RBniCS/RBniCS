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

from math import sqrt
from numpy import isclose
from rbnics.problems.elliptic_optimal_control.elliptic_optimal_control_reduced_problem import EllipticOptimalControlReducedProblem
from rbnics.backends import product, sum, transpose
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.elliptic_optimal_control.elliptic_optimal_control_problem import EllipticOptimalControlProblem
from rbnics.problems.base import LinearRBReducedProblem, ParametrizedReducedDifferentialProblem
from rbnics.reduction_methods.elliptic_optimal_control import EllipticOptimalControlRBReduction

EllipticOptimalControlRBReducedProblem_Base = LinearRBReducedProblem(EllipticOptimalControlReducedProblem(ParametrizedReducedDifferentialProblem))

@ReducedProblemFor(EllipticOptimalControlProblem, EllipticOptimalControlRBReduction)
class EllipticOptimalControlRBReducedProblem(EllipticOptimalControlRBReducedProblem_Base):

    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        EllipticOptimalControlRBReducedProblem_Base.__init__(self, truth_problem, **kwargs)
        
        # Riesz terms names
        self.riesz_terms = ["a", "a*", "c", "c*", "m", "n", "g", "f"]
        self.error_estimation_terms = [("g", "g"), ("f", "f"), ("m", "g"), ("a*", "g"), ("a", "f"), ("c", "f"), ("m", "a*"), ("n", "c*"), ("a", "c"), ("m", "m"), ("a*", "a*"), ("n", "n"), ("c*", "c*"), ("a", "a"), ("c", "c")]
        
    # Return an error bound for the current solution
    def estimate_error(self):
        eps2 = self.get_residual_norm_squared()
        beta = self.truth_problem.get_stability_factor_lower_bound()
        assert eps2 >= 0. or isclose(eps2, 0.)
        assert beta >= 0.
        return sqrt(abs(eps2)/beta)
        
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
        theta_at = self.compute_theta("a*")
        theta_c = self.compute_theta("c")
        theta_ct = self.compute_theta("c*")
        theta_m = self.compute_theta("m")
        theta_n = self.compute_theta("n")
        theta_f = self.compute_theta("f")
        theta_g = self.compute_theta("g")
        
        return (
              sum(product(theta_g, self.error_estimation_operator["g", "g"], theta_g))
            + sum(product(theta_f, self.error_estimation_operator["f", "f"], theta_f))
            + 2.0*(transpose(self._solution)*sum(product(theta_m, self.error_estimation_operator["m", "g"][:N], theta_g)))
            + 2.0*(transpose(self._solution)*sum(product(theta_at, self.error_estimation_operator["a*", "g"][:N], theta_g)))
            + 2.0*(transpose(self._solution)*sum(product(theta_a, self.error_estimation_operator["a", "f"][:N], theta_f)))
            - 2.0*(transpose(self._solution)*sum(product(theta_c, self.error_estimation_operator["c", "f"][:N], theta_f)))
            + transpose(self._solution)*sum(product(theta_m, self.error_estimation_operator["m", "m"][:N, :N], theta_m))*self._solution
            + transpose(self._solution)*sum(product(theta_at, self.error_estimation_operator["a*", "a*"][:N, :N], theta_at))*self._solution
            + 2.0*(transpose(self._solution)*sum(product(theta_m, self.error_estimation_operator["m", "a*"][:N, :N], theta_at))*self._solution)
            + transpose(self._solution)*sum(product(theta_n, self.error_estimation_operator["n", "n"][:N, :N], theta_n))*self._solution
            + transpose(self._solution)*sum(product(theta_ct, self.error_estimation_operator["c*", "c*"][:N, :N], theta_ct))*self._solution
            - 2.0*(transpose(self._solution)*sum(product(theta_n, self.error_estimation_operator["n", "c*"][:N, :N], theta_ct))*self._solution)
            + transpose(self._solution)*sum(product(theta_a, self.error_estimation_operator["a", "a"][:N, :N], theta_a))*self._solution
            + transpose(self._solution)*sum(product(theta_c, self.error_estimation_operator["c", "c"][:N, :N], theta_c))*self._solution
            - 2.0*(transpose(self._solution)*sum(product(theta_a, self.error_estimation_operator["a", "c"][:N, :N], theta_c))*self._solution)
        )
