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
from rbnics.utils.decorators import overload

def compute_theta_for_stability_factor(compute_theta):
    from rbnics.problems.elliptic import EllipticCoerciveProblem
    
    module = types.ModuleType("compute_theta_for_stability_factor", "Storage for implementation of compute_theta_for_stability_factor")
    
    def compute_theta_for_stability_factor_impl(self, term):
        return module._compute_theta_for_stability_factor_impl(self, term)
        
    # Elliptic coercive problem
    @overload(EllipticCoerciveProblem, str, module=module)
    def _compute_theta_for_stability_factor_impl(self_, term):
        if term == "stability_factor_left_hand_matrix":
            return tuple(0.5*t for t in compute_theta(self_, "a"))
        else:
            return compute_theta(self_, term)
    
    return compute_theta_for_stability_factor_impl
