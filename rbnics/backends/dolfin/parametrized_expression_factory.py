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

from ufl.core.operator import Operator
from dolfin import assemble, dx, Expression, Function, FunctionSpace, inner, TensorFunctionSpace, TestFunction, TrialFunction, VectorFunctionSpace
from rbnics.backends.basic import ParametrizedExpressionFactory as BasicParametrizedExpressionFactory
import rbnics.backends.dolfin
from rbnics.utils.decorators import BackendFor, Extends

@Extends(BasicParametrizedExpressionFactory)
@BackendFor("dolfin", inputs=((Expression, Function, Operator), ))
class ParametrizedExpressionFactory(BasicParametrizedExpressionFactory):
    def __init__(self, expression):
        # Extract mesh from expression
        assert isinstance(expression, (Expression, Function, Operator))
        mesh = expression.ufl_domain().ufl_cargo() # from dolfin/fem/projection.py, _extract_function_space function
        # The EIM algorithm will evaluate the expression at vertices. It is thus enough
        # to use a CG1 space.
        shape = expression.ufl_shape
        assert len(shape) in (0, 1, 2)
        if len(shape) == 0:
            space = FunctionSpace(mesh, "Lagrange", 1)
        elif len(shape) == 1:
            space = VectorFunctionSpace(mesh, "Lagrange", 1, dim=shape[0])
        elif len(shape) == 2:
            space = TensorFunctionSpace(mesh, "Lagrange", 1, shape=shape)
        else:
            raise AssertionError("Invalid expression in ParametrizedExpressionFactory.__init__().")
        # Define inner product for POD
        f = TrialFunction(space)
        g = TestFunction(space)
        inner_product = assemble(inner(f, g)*dx)
        # Call Parent
        BasicParametrizedExpressionFactory.__init__(self, expression, space, inner_product, rbnics.backends.dolfin, rbnics.backends.dolfin.wrapping)
                
