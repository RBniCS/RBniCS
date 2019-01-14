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
from rbnics.backends.dolfin.wrapping.assemble_operator_for_restriction import assemble_operator_for_restriction
from rbnics.utils.decorators import overload

def assemble_operator_for_supremizers(assemble_operator):
    from rbnics.problems.stokes import StokesProblem
    from rbnics.problems.stokes_optimal_control import StokesOptimalControlProblem
    
    module = types.ModuleType("assemble_operator_for_supremizers", "Storage for implementation of assemble_operator_for_supremizers")
    
    def assemble_operator_for_supremizers_impl(self, term):
        return module._assemble_operator_for_supremizers_impl(self, term)
        
    # Stokes problem
    @overload(StokesProblem, str, module=module)
    def _assemble_operator_for_supremizers_impl(self_, term):
        return _assemble_operator_for_supremizers_impl_stokes_problem(self_, term)
        
    _assemble_operator_for_supremizers_impl_stokes_problem = (
        assemble_operator_for_restriction({"bt_restricted": "bt"}, test="s")(
        assemble_operator_for_restriction({"dirichlet_bc_s": "dirichlet_bc_u"}, trial="s")(
        assemble_operator_for_restriction({"inner_product_s": "inner_product_u"}, test="s", trial="s")(
            assemble_operator
        )
        )
        )
    )
    
    # Stokes optimal control problem
    @overload(StokesOptimalControlProblem, str, module=module)
    def _assemble_operator_for_supremizers_impl(self_, term):
        return _assemble_operator_for_supremizers_impl_stokes_optimal_control_problem(self_, term)
        
    _assemble_operator_for_supremizers_impl_stokes_optimal_control_problem = (
        assemble_operator_for_restriction({"bt*_restricted": "bt*"}, test="s")(
        assemble_operator_for_restriction({"bt_restricted": "bt"}, test="r")(
        assemble_operator_for_restriction({"dirichlet_bc_s": "dirichlet_bc_v"}, trial="s")(
        assemble_operator_for_restriction({"dirichlet_bc_r": "dirichlet_bc_w"}, trial="r")(
        assemble_operator_for_restriction({"inner_product_s": "inner_product_v"}, test="s", trial="s")(
        assemble_operator_for_restriction({"inner_product_r": "inner_product_w"}, test="r", trial="r")(
            assemble_operator
        )
        )
        )
        )
        )
        )
    )
    
    return assemble_operator_for_supremizers_impl
