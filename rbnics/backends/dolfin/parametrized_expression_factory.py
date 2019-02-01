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

from ufl.core.operator import Operator
from ufl.domain import extract_domains
from ufl.corealg.traversal import traverse_unique_terminals
from dolfin import assemble, dx, FunctionSpace, inner, TensorFunctionSpace, TestFunction, TrialFunction, VectorFunctionSpace
from dolfin.function.expression import BaseExpression
from rbnics.backends.basic import ParametrizedExpressionFactory as BasicParametrizedExpressionFactory
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.functions_list import FunctionsList
from rbnics.backends.dolfin.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from rbnics.backends.dolfin.reduced_vertices import ReducedVertices
from rbnics.backends.dolfin.snapshots_matrix import SnapshotsMatrix
from rbnics.backends.dolfin.wrapping import expression_description, expression_iterator, expression_name, get_auxiliary_problem_for_non_parametrized_function, is_parametrized, is_problem_solution, is_problem_solution_dot, is_problem_solution_type, is_time_dependent, solution_dot_identify_component, solution_identify_component, solution_iterator
from rbnics.utils.decorators import BackendFor, ModuleWrapper, overload

backend = ModuleWrapper(Function, FunctionsList, ProperOrthogonalDecomposition, ReducedVertices, SnapshotsMatrix)
wrapping = ModuleWrapper(expression_iterator, is_problem_solution, is_problem_solution_dot, is_problem_solution_type, solution_dot_identify_component, solution_identify_component, solution_iterator, expression_description=expression_description, expression_name=expression_name, get_auxiliary_problem_for_non_parametrized_function=get_auxiliary_problem_for_non_parametrized_function, is_parametrized=is_parametrized, is_time_dependent=is_time_dependent)
ParametrizedExpressionFactory_Base = BasicParametrizedExpressionFactory(backend, wrapping)

@BackendFor("dolfin", inputs=((BaseExpression, Function.Type(), Operator), ))
class ParametrizedExpressionFactory(ParametrizedExpressionFactory_Base):
    def __init__(self, expression):
        # Generate space
        space = _generate_space(expression)
        # Define inner product for POD
        f = TrialFunction(space)
        g = TestFunction(space)
        inner_product = assemble(inner(f, g)*dx)
        # Call Parent
        ParametrizedExpressionFactory_Base.__init__(self, expression, space, inner_product)

# Space generation for BaseExpression
@overload
def _generate_space(expression: BaseExpression):
    # Extract mesh from expression
    assert hasattr(expression, "_mesh")
    mesh = expression._mesh
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
        raise ValueError("Invalid expression in ParametrizedExpressionFactory.__init__().")
    return space

# Space generation for Function
@overload
def _generate_space(expression: Function.Type()):
    return expression.function_space()

# Space generation for Operator
@overload
def _generate_space(expression: Operator):
    # Extract mesh from expression
    meshes = set([ufl_domain.ufl_cargo() for ufl_domain in extract_domains(expression)]) # from dolfin/fem/projection.py, _extract_function_space function
    for t in traverse_unique_terminals(expression): # from ufl/domain.py, extract_domains
        if hasattr(t, "_mesh"):
            meshes.add(t._mesh)
    assert len(meshes) == 1
    mesh = meshes.pop()
    # The EIM algorithm will evaluate the expression at vertices. However, since the Operator expression may
    # contain e.g. a gradient of a solution defined in a C^0 space, we resort to DG1 spaces.
    shape = expression.ufl_shape
    assert len(shape) in (0, 1, 2)
    if len(shape) == 0:
        space = FunctionSpace(mesh, "Discontinuous Lagrange", 1)
    elif len(shape) == 1:
        space = VectorFunctionSpace(mesh, "Discontinuous Lagrange", 1, dim=shape[0])
    elif len(shape) == 2:
        space = TensorFunctionSpace(mesh, "Discontinuous Lagrange", 1, shape=shape)
    else:
        raise ValueError("Invalid expression in ParametrizedExpressionFactory.__init__().")
    return space
