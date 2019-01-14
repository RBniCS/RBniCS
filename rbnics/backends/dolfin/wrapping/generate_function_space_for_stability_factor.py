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
from dolfin import FunctionSpace
from rbnics.utils.decorators import overload

def generate_function_space_for_stability_factor(__init__):
    from rbnics.problems.elliptic import EllipticCoerciveProblem
    from rbnics.problems.stokes import StokesProblem
    
    module = types.ModuleType("generate_function_space_for_stability_factor", "Storage for implementation of generate_function_space_for_stability_factor")
    
    def generate_function_space_for_stability_factor_impl(self, V, **kwargs):
        __init__(self, V, **kwargs)
        module._generate_function_space_for_stability_factor_impl(self, V)
        
    # Elliptic coercive or Stokes problems
    @overload((EllipticCoerciveProblem, StokesProblem), FunctionSpace, module=module)
    def _generate_function_space_for_stability_factor_impl(self_, V):
        self_.stability_factor_V = V
    
    return generate_function_space_for_stability_factor_impl
