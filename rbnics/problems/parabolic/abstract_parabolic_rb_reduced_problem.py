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
from rbnics.problems.base import LinearTimeDependentRBReducedProblem
from rbnics.backends import assign, product, sum, TimeSeries, transpose

def AbstractParabolicRBReducedProblem(AbstractParabolicReducedProblem_DerivedClass):
    AbstractParabolicRBReducedProblem_Base = LinearTimeDependentRBReducedProblem(AbstractParabolicReducedProblem_DerivedClass)

    class AbstractParabolicRBReducedProblem_Class(AbstractParabolicRBReducedProblem_Base):
        
        # Default initialization of members.
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            AbstractParabolicRBReducedProblem_Base.__init__(self, truth_problem, **kwargs)
            
            # Skip useless Riesz products
            self.riesz_terms.append("m")
            self.error_estimation_terms.extend([("m", "f"), ("m", "a"), ("m", "m")])
            
        # Return an error bound for the current solution
        def estimate_error(self):
            eps2_over_time = self.get_residual_norm_squared()
            beta = self.truth_problem.get_stability_factor_lower_bound()
            # Compute error bound
            error_bound_over_time = TimeSeries(eps2_over_time)
            for (k, t) in enumerate(eps2_over_time.stored_times()):
                if not isclose(t, self.t0, self.dt/2.):
                    eps2 = eps2_over_time[k]
                    assert eps2 >= 0. or isclose(eps2, 0.)
                    assert beta >= 0.
                    error_bound_over_time.append(sqrt(abs(eps2)/beta))
                else:
                    initial_error_estimate_squared = self.get_initial_error_estimate_squared()
                    assert initial_error_estimate_squared >= 0. or isclose(initial_error_estimate_squared, 0.)
                    error_bound_over_time.append(sqrt(abs(initial_error_estimate_squared)))
            #
            return error_bound_over_time
            
        # Return an error bound for the current solution
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
            residual_norm_squared_over_time = TimeSeries(self._solution_over_time)
            assert len(self._solution_over_time) == len(self._solution_dot_over_time)
            for (k, t) in enumerate(self._solution_over_time.stored_times()):
                if not isclose(t, self.t0, self.dt/2.):
                    # Set current time
                    self.set_time(t)
                    # Set current solution and solution_dot
                    assign(self._solution, self._solution_over_time[k])
                    assign(self._solution_dot, self._solution_dot_over_time[k])
                    # Compute the numerator of the error bound at the current time, first
                    # by computing residual of elliptic part
                    elliptic_residual_norm_squared = AbstractParabolicRBReducedProblem_Base.get_residual_norm_squared(self)
                    # ... and then adding terms related to time derivative
                    N = self._solution.N
                    theta_m = self.compute_theta("m")
                    theta_a = self.compute_theta("a")
                    theta_f = self.compute_theta("f")
                    residual_norm_squared_over_time.append(
                          elliptic_residual_norm_squared
                        + 2.0*(transpose(self._solution_dot)*sum(product(theta_m, self.error_estimation_operator["m", "f"][:N], theta_f)))
                        + 2.0*(transpose(self._solution_dot)*sum(product(theta_m, self.error_estimation_operator["m", "a"][:N, :N], theta_a))*self._solution)
                        + transpose(self._solution_dot)*sum(product(theta_m, self.error_estimation_operator["m", "m"][:N, :N], theta_m))*self._solution_dot
                    )
                else:
                    # Error estimator on initial condition does not use the residual
                    residual_norm_squared_over_time.append(0.)
            return residual_norm_squared_over_time
            
    # return value (a class) for the decorator
    return AbstractParabolicRBReducedProblem_Class
