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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl.algorithms.traversal import iter_expressions
from ufl.core.operator import Operator
from ufl.corealg.traversal import traverse_unique_terminals
from dolfin import assemble, dx, Expression, Function, FunctionSpace, inner, TestFunction, TrialFunction
from RBniCS.backends.abstract import ParametrizedExpressionFactory as AbstractParametrizedExpressionFactory
from RBniCS.backends.fenics.functions_list import FunctionsList
from RBniCS.backends.fenics.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from RBniCS.backends.fenics.reduced_mesh import ReducedMesh
from RBniCS.backends.fenics.reduced_vertices import ReducedVertices
from RBniCS.backends.fenics.snapshots_matrix import SnapshotsMatrix
from RBniCS.backends.fenics.wrapping import function_from_subfunction_if_any, function_space_for_expression_projection, get_expression_description, get_expression_name
from RBniCS.utils.decorators import BackendFor, Extends, get_problem_from_solution, get_reduced_problem_from_problem, override
from RBniCS.utils.mpi import parallel_max

@Extends(AbstractParametrizedExpressionFactory)
@BackendFor("fenics", inputs=(object, (Expression, Function, Operator))) # object will actually be a ParametrizedDifferentialProblem
class ParametrizedExpressionFactory(AbstractParametrizedExpressionFactory):
    def __init__(self, truth_problem, expression):
        AbstractParametrizedExpressionFactory.__init__(self, truth_problem, expression)
        self._truth_problem = truth_problem
        self._expression = expression
        self._name = get_expression_name(expression)
        assert isinstance(expression, (Expression, Function, Operator))
        if isinstance(expression, Expression):
            self._space = FunctionSpace(expression.mesh, expression.ufl_element())
        elif isinstance(expression, Function):
            self._space = expression.function_space()
        elif isinstance(expression, Operator):
            self._space = function_space_for_expression_projection(expression)
        else:
            raise AssertionError("Invalid expression in ParametrizedExpressionFactory.__init__().")
            
    @override
    def create_interpolation_locations_container(self):
        return ReducedVertices(self._space)
        
    @override
    def create_snapshots_container(self):
        return SnapshotsMatrix(self._space)
        
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
    def description(self):
        return PrettyTuple(self._expression, get_expression_description(self._expression), self._name)
        
    @override
    def is_nonlinear(self):
        visited = list()
        all_truth_problems = list()
        
        # Look for terminals on truth mesh
        for subexpression in iter_expressions(self._expression):
            for node in traverse_unique_terminals(subexpression):
                node = function_from_subfunction_if_any(node)
                if node in visited:
                    continue
                # ... problem solutions related to nonlinear terms
                elif isinstance(node, Function):
                    truth_problem = get_problem_from_solution(node)
                    all_truth_problems.append(truth_problem)
                    
        return self._truth_problem in all_truth_problems
        
class PrettyTuple(tuple):
    def __new__(cls, arg0, arg1, arg2):
        as_list = [str(arg0) + ",", "where"]
        as_list.extend([str(key) + " = " + value for key, value in arg1.iteritems()])
        as_list.append("with id " + str(arg2))
        return tuple.__new__(cls, tuple(as_list))
        
