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

from ufl.core.operator import Operator
from dolfin import assemble, dx, Expression, FunctionSpace, inner, Point, project, TensorElement, TestFunction, TrialFunction, VectorElement
from RBniCS.backends.abstract import ParametrizedExpressionFactory as AbstractParametrizedExpressionFactory
from RBniCS.backends.fenics.functions_list import FunctionsList
from RBniCS.backends.fenics.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from RBniCS.backends.fenics.reduced_mesh import ReducedMesh
from RBniCS.backends.fenics.reduced_vertices import ReducedVertices
from RBniCS.backends.fenics.snapshots_matrix import SnapshotsMatrix
from RBniCS.utils.decorators import BackendFor, Extends, override
from RBniCS.utils.mpi import parallel_max

@Extends(AbstractParametrizedExpressionFactory)
@BackendFor("fenics", inputs=((Expression, Operator), ))
class ParametrizedExpressionFactory(AbstractParametrizedExpressionFactory):
    def __init__(self, expression):
        AbstractParametrizedExpressionFactory.__init__(self, expression)
        assert isinstance(expression, (Expression, Operator))
        if isinstance(expression, Expression):
            self._expression = expression
            self._space = FunctionSpace(expression.mesh, expression.ufl_element())
        elif isinstance(expression, Operator):
            self._expression = expression
            self._space = project(expression).function_space() # automatically determines the FunctionSpace
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
        
