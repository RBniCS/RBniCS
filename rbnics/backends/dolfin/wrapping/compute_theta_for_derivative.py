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

def compute_theta_for_derivative(jacobian_term_to_residual_term):
    def compute_theta_for_derivative_decorator(compute_theta):
        def compute_theta_for_derivative_decorator_impl(self, term):
            residual_term = jacobian_term_to_residual_term.get(term)
            if residual_term is None: # term was not a jacobian_term
                return compute_theta(self, term)
            else:
                return compute_theta(self, residual_term)
            
        return compute_theta_for_derivative_decorator_impl
    return compute_theta_for_derivative_decorator
