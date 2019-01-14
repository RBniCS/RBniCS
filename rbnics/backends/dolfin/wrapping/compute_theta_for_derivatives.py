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

import types
from rbnics.backends.dolfin.wrapping.compute_theta_for_derivative import compute_theta_for_derivative
from rbnics.utils.decorators import overload

def compute_theta_for_derivatives(compute_theta):
    from rbnics.problems.nonlinear_elliptic import NonlinearEllipticProblem
    from rbnics.problems.navier_stokes import NavierStokesProblem
    
    module = types.ModuleType("compute_theta_for_derivatives", "Storage for implementation of compute_theta_for_derivatives")
    
    def compute_theta_for_derivatives_impl(self, term):
        return module._compute_theta_for_derivatives_impl(self, term)
        
    # Nonlinear elliptic problem
    @overload(NonlinearEllipticProblem, str, module=module)
    def _compute_theta_for_derivatives_impl(self_, term):
        return _compute_theta_for_derivatives_impl_nonlinear_elliptic_problem(self_, term)
        
    _compute_theta_for_derivatives_impl_nonlinear_elliptic_problem = (
        compute_theta_for_derivative({"dc": "c"})(
            compute_theta
        )
    )
    
    # Navier Stokes problem
    @overload(NavierStokesProblem, str, module=module)
    def _compute_theta_for_derivatives_impl(self_, term):
        return _compute_theta_for_derivatives_impl_navier_stokes_problem(self_, term)
        
    _compute_theta_for_derivatives_impl_navier_stokes_problem = (
        compute_theta_for_derivative({"dc": "c"})(
            compute_theta
        )
    )
    
    return compute_theta_for_derivatives_impl
