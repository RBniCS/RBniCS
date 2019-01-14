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
from rbnics.problems.stokes.stokes_reduced_problem import StokesReducedProblem
from rbnics.backends import product, sum, transpose
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.stokes.stokes_problem import StokesProblem
from rbnics.problems.base import LinearRBReducedProblem, ParametrizedReducedDifferentialProblem
from rbnics.reduction_methods.stokes import StokesRBReduction

StokesRBReducedProblem_Base = LinearRBReducedProblem(StokesReducedProblem(ParametrizedReducedDifferentialProblem))

@ReducedProblemFor(StokesProblem, StokesRBReduction)
class StokesRBReducedProblem(StokesRBReducedProblem_Base):

    # Default initialization of members.
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        StokesRBReducedProblem_Base.__init__(self, truth_problem, **kwargs)
        
        # Skip useless Riesz products
        self.riesz_terms = ["a", "b", "bt", "f", "g"]
        self.error_estimation_terms = [("f", "f"), ("g", "g"), ("a", "f"), ("bt", "f"), ("b", "g"), ("a", "a"), ("a", "bt"), ("bt", "bt"), ("b", "b")]
    
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
        theta_b = self.compute_theta("b")
        theta_bt = self.compute_theta("bt")
        theta_f = self.compute_theta("f")
        theta_g = self.compute_theta("g")
        return (
              sum(product(theta_f, self.error_estimation_operator["f", "f"], theta_f))
            + sum(product(theta_g, self.error_estimation_operator["g", "g"], theta_g))
            + 2.0*(transpose(self._solution)*sum(product(theta_a, self.error_estimation_operator["a", "f"][:N], theta_f)))
            + 2.0*(transpose(self._solution)*sum(product(theta_bt, self.error_estimation_operator["bt", "f"][:N], theta_f)))
            + 2.0*(transpose(self._solution)*sum(product(theta_b, self.error_estimation_operator["b", "g"][:N], theta_g)))
            + transpose(self._solution)*sum(product(theta_a, self.error_estimation_operator["a", "a"][:N, :N], theta_a))*self._solution
            + 2.0*(transpose(self._solution)*sum(product(theta_a, self.error_estimation_operator["a", "bt"][:N, :N], theta_bt))*self._solution)
            + transpose(self._solution)*sum(product(theta_bt, self.error_estimation_operator["bt", "bt"][:N, :N], theta_bt))*self._solution
            + transpose(self._solution)*sum(product(theta_b, self.error_estimation_operator["b", "b"][:N, :N], theta_b))*self._solution
        )
