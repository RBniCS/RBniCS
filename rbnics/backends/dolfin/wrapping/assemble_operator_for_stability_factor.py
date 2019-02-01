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
from numpy import zeros
from dolfin import adjoint, Constant
from rbnics.backends.dolfin.wrapping.dirichlet_bc import DirichletBC
from rbnics.utils.decorators import overload

def assemble_operator_for_stability_factor(assemble_operator):
    from rbnics.problems.elliptic import EllipticCoerciveProblem
    
    module = types.ModuleType("assemble_operator_for_stability_factor", "Storage for implementation of assemble_operator_for_stability_factor")
    
    def assemble_operator_for_stability_factor_impl(self, term):
        return module._assemble_operator_for_stability_factor_impl(self, term)
        
    # Elliptic coercive problem
    @overload(EllipticCoerciveProblem, str, module=module)
    def _assemble_operator_for_stability_factor_impl(self_, term):
        if term == "stability_factor_left_hand_matrix":
            return tuple(f + adjoint(f) for f in assemble_operator(self_, "a"))
        elif term == "stability_factor_right_hand_matrix":
            return assemble_operator(self_, "inner_product")
        elif term == "stability_factor_dirichlet_bc":
            original_dirichlet_bcs = assemble_operator(self_, "dirichlet_bc")
            zeroed_dirichlet_bcs = list()
            for original_dirichlet_bc in original_dirichlet_bcs:
                zeroed_dirichlet_bc = list()
                for original_dirichlet_bc_i in original_dirichlet_bc:
                    args = list()
                    args.append(original_dirichlet_bc_i.function_space())
                    zero_value = Constant(zeros(original_dirichlet_bc_i.value().ufl_shape))
                    args.append(zero_value)
                    args.extend(original_dirichlet_bc_i._domain)
                    kwargs = original_dirichlet_bc_i._kwargs
                    zeroed_dirichlet_bc.append(DirichletBC(*args, **kwargs))
                assert len(zeroed_dirichlet_bc) == len(original_dirichlet_bc)
                zeroed_dirichlet_bcs.append(zeroed_dirichlet_bc)
            assert len(zeroed_dirichlet_bcs) == len(original_dirichlet_bcs)
            return tuple(zeroed_dirichlet_bcs)
        else:
            return assemble_operator(self_, term)
    
    return assemble_operator_for_stability_factor_impl
