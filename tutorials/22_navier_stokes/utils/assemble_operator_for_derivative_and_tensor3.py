# Copyright (C) 2015-2017 by the RBniCS authors
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

from ufl.algorithms import expand_derivatives
from dolfin import derivative, split, TestFunction, TrialFunction
from rbnics.backends.dolfin.wrapping.expression_replace import replace

def assemble_operator_for_derivative_and_tensor3(jacobian_term_to_residual_term):
    assert len(jacobian_term_to_residual_term)
    assert "dc" in jacobian_term_to_residual_term
    assert jacobian_term_to_residual_term["dc"] == "c"
    
    def assemble_operator_for_derivative_and_tensor3_decorator(assemble_operator):
        def assemble_operator_for_derivative_and_tensor3_decorator_impl(self, term):
            if term in ("c", "c_tensor3", "dc", "dc_tensor3"):
                c_tensor3_operator = assemble_operator(self, "c")
                if term == "c_tensor3":
                    return c_tensor3_operator
                else:
                    (u, _) = split(self._solution)
                    (v, _) = split(TestFunction(self.V))
                    (u_placeholder_1, _) = split(self._solution_placeholder_1)
                    (u_placeholder_2, _) = split(self._solution_placeholder_2)
                    (u_placeholder_3, _) = split(self._solution_placeholder_3)
                    c_operator = tuple(replace(c_tensor3_op, {u_placeholder_1: u, u_placeholder_2: u, u_placeholder_3: v}) for c_tensor3_op in c_tensor3_operator)
                    if term == "c":
                        return c_operator
                    else:
                        (du, _) = split(TrialFunction(self.V))
                        dc_operator = tuple(expand_derivatives(derivative(c_op, u, du)) for c_op in c_operator)
                        if term == "dc":
                            return dc_operator
                        else:
                            dc_tensor3_operator = tuple(replace(dc_op, {u: u_placeholder_1, du: u_placeholder_2, v: u_placeholder_3}) for dc_op in dc_operator)
                            return dc_tensor3_operator
            else:
                return assemble_operator(self, term)
            
        return assemble_operator_for_derivative_and_tensor3_decorator_impl
    return assemble_operator_for_derivative_and_tensor3_decorator
