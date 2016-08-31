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

from dolfin import Expression, FunctionSpace, Point, TensorFunctionSpace, VectorFunctionSpace
from RBniCS.backends.abstract import ProjectedParametrizedExpression as AbstractProjectedParametrizedExpression
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
        element = original_space.ufl_element()
        assert len(shape) in (0, 1, 2)
        if len(shape) == 0:
            pass
        elif len(shape) == 1:
            element = VectorElement(element)
        elif len(shape) == 2:
            element = TensorElement(element)
        else:
            raise AssertionError("Invalid function space in ProjectedParametrizedExpression")
        self._space = FunctionSpace(original_space.mesh(), element)
        mesh = self._space.mesh()
        self._bounding_box_tree = mesh.bounding_box_tree()
        self._mpi_comm = mesh.mpi_comm().tompi4py()
    
    @override
    @property
    def expression(self):
        return self._expression
        
    @override
    @property
    def space(self):
        return self._space
        
    @override
    def get_processor_id(self, point):
        is_local = self._bounding_box_tree.collides_entity(Point(point))
        processor_id = -1
        if is_local:
            processor_id = self._mpi_comm.rank
        return parallel_max(self._mpi_comm, processor_id)
        
