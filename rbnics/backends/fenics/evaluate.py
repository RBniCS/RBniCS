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

from rbnics.backends.basic import evaluate as basic_evaluate
import rbnics.backends.fenics
from rbnics.backends.fenics.function import Function
from rbnics.backends.fenics.functions_list import FunctionsList
from rbnics.backends.fenics.matrix import Matrix
from rbnics.backends.fenics.parametrized_expression_factory import ParametrizedExpressionFactory
from rbnics.backends.fenics.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.fenics.reduced_mesh import ReducedMesh
from rbnics.backends.fenics.reduced_vertices import ReducedVertices
from rbnics.backends.fenics.tensors_list import TensorsList
from rbnics.backends.fenics.vector import Vector
from rbnics.utils.decorators import backend_for

# Evaluate a parametrized expression, possibly at a specific location
@backend_for("fenics", inputs=((Matrix.Type(), Vector.Type(), Function.Type(), TensorsList, FunctionsList, ParametrizedTensorFactory, ParametrizedExpressionFactory), (ReducedMesh, ReducedVertices, None)))
def evaluate(expression, at=None):
    return basic_evaluate(expression, at, rbnics.backends.fenics, rbnics.backends.fenics.wrapping)
