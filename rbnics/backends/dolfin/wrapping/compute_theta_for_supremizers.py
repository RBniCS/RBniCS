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
from rbnics.backends.dolfin.wrapping.compute_theta_for_restriction import compute_theta_for_restriction
from rbnics.utils.decorators import overload

def compute_theta_for_supremizers(compute_theta):
    from rbnics.problems.stokes import StokesProblem
    from rbnics.problems.stokes_optimal_control import StokesOptimalControlProblem
    
    module = types.ModuleType("compute_theta_for_supremizers", "Storage for implementation of compute_theta_for_supremizers")
    
    def compute_theta_for_supremizers_impl(self, term):
        return module._compute_theta_for_supremizers_impl(self, term)
        
    # Stokes problem
    @overload(StokesProblem, str, module=module)
    def _compute_theta_for_supremizers_impl(self_, term):
        return _compute_theta_for_supremizers_impl_stokes_problem(self_, term)
        
    _compute_theta_for_supremizers_impl_stokes_problem = (
        compute_theta_for_restriction({"bt_restricted": "bt"})(
            compute_theta
        )
    )
    
    # Stokes optimal control problem
    @overload(StokesOptimalControlProblem, str, module=module)
    def _compute_theta_for_supremizers_impl(self_, term):
        return _compute_theta_for_supremizers_impl_stokes_optimal_control_problem(self_, term)
        
    _compute_theta_for_supremizers_impl_stokes_optimal_control_problem = (
        compute_theta_for_restriction({"bt*_restricted": "bt*"})(
        compute_theta_for_restriction({"bt_restricted": "bt"})(
            compute_theta
        )
        )
    )
    
    return compute_theta_for_supremizers_impl
