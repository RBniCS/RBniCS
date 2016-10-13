# Copyright (C) 2015-2016 by the RBniCS authors
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

from dolfin import assemble, dx, Expression, FunctionSpace, inner, Point, TensorElement, TestFunction, TrialFunction, VectorElement
from RBniCS.backends.abstract import ProjectedParametrizedExpression as AbstractProjectedParametrizedExpression
from RBniCS.backends.fenics.functions_list import FunctionsList
from RBniCS.backends.fenics.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from RBniCS.backends.fenics.reduced_vertices import ReducedVertices
from RBniCS.backends.fenics.snapshots_matrix import SnapshotsMatrix
from RBniCS.utils.decorators import BackendFor, Extends, override
from RBniCS.utils.mpi import parallel_max

@Extends(AbstractProjectedParametrizedExpression)
@BackendFor("FEniCS", inputs=(Expression, FunctionSpace))
class ProjectedParametrizedExpression(AbstractProjectedParametrizedExpression):
    def __init__(self, expression, original_space):
        AbstractProjectedParametrizedExpression.__init__(self, expression, original_space)
        #
        self._expression = expression
        shape = expression.ufl_shape
        sub_elements_queue = original_space.ufl_element().sub_elements()
        if len(sub_elements_queue) > 0:
            sub_elements_queue_index = 0
            scalar_sub_elements = list()
            while sub_elements_queue_index < len(sub_elements_queue):
                current_sub_elements = sub_elements_queue[sub_elements_queue_index].sub_elements()
                if len(current_sub_elements) > 0:
                    for current_sub_element in current_sub_elements:
                        sub_elements_queue.append(current_sub_element)
                else:
                    scalar_sub_elements.append(sub_elements_queue[sub_elements_queue_index])
                sub_elements_queue_index += 1
            assert len(scalar_sub_elements) > 0
            first_family = scalar_sub_elements[0].family()
            max_degree = scalar_sub_elements[0].degree()
            scalar_element = scalar_sub_elements[0]
            for scalar_sub_element in scalar_sub_elements:
                assert scalar_sub_element.family() == first_family
                if scalar_sub_element.degree() > max_degree:
                    scalar_element = scalar_sub_element
        else:
            scalar_element = original_space.ufl_element()
        assert len(shape) in (0, 1, 2)
        if len(shape) == 0:
            expression_element = scalar_element
        elif len(shape) == 1:
            expression_element = VectorElement(scalar_element)
        elif len(shape) == 2:
            expression_element = TensorElement(scalar_element)
        else:
            raise AssertionError("Invalid function space in ProjectedParametrizedExpression")
        self._space = FunctionSpace(original_space.mesh(), expression_element)
            
    @override
    def create_interpolation_locations_container(self):
        return ReducedVertices(self._space.mesh())
        
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
        
