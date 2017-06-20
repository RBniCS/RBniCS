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

from ufl.algorithms.traversal import iter_expressions
from ufl.core.operator import Operator
from ufl.corealg.traversal import traverse_unique_terminals
from dolfin import assemble, dx, Expression, Function, FunctionSpace, inner, TensorFunctionSpace, TestFunction, TrialFunction, VectorFunctionSpace
from rbnics.backends.abstract import ParametrizedExpressionFactory as AbstractParametrizedExpressionFactory
from rbnics.backends.fenics.functions_list import FunctionsList
from rbnics.backends.fenics.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from rbnics.backends.fenics.reduced_mesh import ReducedMesh
from rbnics.backends.fenics.reduced_vertices import ReducedVertices
from rbnics.backends.fenics.snapshots_matrix import SnapshotsMatrix
from rbnics.backends.fenics.wrapping import function_from_subfunction_if_any, get_expression_description, get_expression_name
from rbnics.utils.decorators import BackendFor, Extends, get_problem_from_solution, get_reduced_problem_from_problem, is_problem_solution, override
from rbnics.utils.mpi import parallel_max

@Extends(AbstractParametrizedExpressionFactory)
@BackendFor("fenics", inputs=((Expression, Function, Operator), ))
class ParametrizedExpressionFactory(AbstractParametrizedExpressionFactory):
    def __init__(self, expression):
        AbstractParametrizedExpressionFactory.__init__(self, expression)
        self._expression = expression
        self._name = get_expression_name(expression)
        # Extract mesh from expression
        assert isinstance(expression, (Expression, Function, Operator))
        mesh = expression.ufl_domain().ufl_cargo() # from dolfin/fem/projection.py, _extract_function_space function
        # The EIM algorithm will evaluate the expression at vertices. It is thus enough
        # to use a CG1 space.
        shape = expression.ufl_shape
        assert len(shape) in (0, 1, 2)
        if len(shape) == 0:
            self._space = FunctionSpace(mesh, "Lagrange", 1)
        elif len(shape) == 1:
            self._space = VectorFunctionSpace(mesh, "Lagrange", 1, dim=shape[0])
        elif len(shape) == 2:
            self._space = TensorFunctionSpace(mesh, "Lagrange", 1, shape=shape)
        else:
            raise AssertionError("Invalid expression in ParametrizedExpressionFactory.__init__().")
            
    @override
    def create_interpolation_locations_container(self):
        return ReducedVertices(self._space)
        
    @override
    def create_snapshots_container(self):
        return SnapshotsMatrix(self._space)
        
    @override
    def create_empty_snapshot(self):
        return Function(self._space)
        
    @override
    def create_basis_container(self):
        # We use FunctionsList instead of BasisFunctionsMatrix since we are not interested in storing multiple components
        return FunctionsList(self._space)
        
    @override
    def create_POD_container(self):
        f = TrialFunction(self._space)
        g = TestFunction(self._space)
        inner_product = assemble(inner(f, g)*dx)
        return ProperOrthogonalDecomposition(self._space, inner_product)
        
    @override
    def name(self):
        return self._name
        
    @override
    def description(self):
        return PrettyTuple(self._expression, get_expression_description(self._expression), self._name)
        
    @override
    def is_parametrized(self):
        if self.is_time_dependent():
            return True
        for subexpression in iter_expressions(self._expression):
            for node in traverse_unique_terminals(subexpression):
                node = function_from_subfunction_if_any(node)
                # ... parametrized expressions
                if isinstance(node, Expression) and "mu_0" in node.user_parameters:
                    return True
                # ... problem solutions related to nonlinear terms
                elif isinstance(node, Function) and is_problem_solution(node):
                    return True
        return False
        
    @override
    def is_time_dependent(self):
        for subexpression in iter_expressions(self._expression):
            for node in traverse_unique_terminals(subexpression):
                node = function_from_subfunction_if_any(node)
                # ... parametrized expressions
                if isinstance(node, Expression) and "t" in node.user_parameters:
                    return True
                # ... problem solutions related to nonlinear terms
                elif isinstance(node, Function) and is_problem_solution(node):
                    truth_problem = get_problem_from_solution(node)
                    if hasattr(truth_problem, "set_time"):
                        return True
        return False
        
class PrettyTuple(tuple):
    def __new__(cls, arg0, arg1, arg2):
        as_list = [str(arg0) + ",", "where"]
        as_list.extend([str(key) + " = " + value for key, value in arg1.iteritems()])
        as_list.append("with id " + str(arg2))
        return tuple.__new__(cls, tuple(as_list))
        
