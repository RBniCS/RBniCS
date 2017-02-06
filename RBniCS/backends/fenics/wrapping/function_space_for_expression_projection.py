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
## @file
#  @brief
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import traverse_unique_terminals
from dolfin import Function, FunctionSpace, TensorFunctionSpace, VectorFunctionSpace
from RBniCS.backends.fenics.wrapping.function_from_subfunction_if_any import function_from_subfunction_if_any

def function_space_for_expression_projection(expression):
    # Get mesh from expression
    mesh = expression.ufl_domain().ufl_cargo() # from dolfin/fem/projection.py, _extract_function_space function
    
    # Get shape
    shape = expression.ufl_shape
    
    # List all solutions related to nonlinear terms, in order to interpolate them with the appropriate
    # function space
    solutions = list()
    
    for subexpression in iter_expressions(expression):
        for node in traverse_unique_terminals(subexpression):
            node = function_from_subfunction_if_any(node)
            if node in solutions:
                continue
            # ... problem solutions related to nonlinear terms
            elif isinstance(node, Function):
                solutions.append(node)
        
    # Get maximum degree of solutions' function spaces
    if len(solutions) == 0:
        max_degree = 1
    else:
        max_degree = -1
        for solution in solutions:
            if solution.function_space().ufl_element().family() == "Mixed":
                for i in range(solution.function_space().num_sub_spaces()):
                    assert solution.function_space().sub(i).ufl_element().family() == "Lagrange", "The current implementation relies on taking the maximum degree, interpolation between different function spaces has not been implemented yet"
                    max_degree = max(solution.function_space().sub(i).ufl_element().degree(), max_degree)
            else:
                assert solution.function_space().ufl_element().family() == "Lagrange", "The current implementation relies on taking the maximum degree, interpolation between different function spaces has not been implemented yet"
                max_degree = max(solution.function_space().ufl_element().degree(), max_degree)
        assert max_degree > 0
    
    # Return a (Lagrange, see asserts) FunctionSpace of suitable degree
    # (from dolfin/fem/projection.py, _extract_function_space function)
    if shape == ():
        return FunctionSpace(mesh, "Lagrange", max_degree)
    elif len(shape) == 1:
        return VectorFunctionSpace(mesh, "Lagrange", max_degree, dim=shape[0])
    elif len(shape) == 2:
        return TensorFunctionSpace(mesh, "Lagrange", max_degree, shape=shape)

