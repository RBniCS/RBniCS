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

from rbnics.backends.abstract import ParametrizedExpressionFactory as AbstractParametrizedExpressionFactory
from rbnics.utils.decorators import Extends, override

@Extends(AbstractParametrizedExpressionFactory)
class ParametrizedExpressionFactory(AbstractParametrizedExpressionFactory):
    def __init__(self, expression, space, inner_product, backend, wrapping):
        AbstractParametrizedExpressionFactory.__init__(self, expression)
        self._expression = expression
        self._name = wrapping.expression_name(expression)
        self._description = PrettyTuple(self._expression, wrapping.expression_description(self._expression), self._name)
        self._space = space
        self._inner_product = inner_product
        self.backend = backend
        self.wrapping = wrapping
            
    @override
    def create_interpolation_locations_container(self):
        return self.backend.ReducedVertices(self._space)
        
    @override
    def create_snapshots_container(self):
        return self.backend.SnapshotsMatrix(self._space)
        
    @override
    def create_empty_snapshot(self):
        return self.backend.Function(self._space)
        
    @override
    def create_basis_container(self):
        # We use FunctionsList instead of BasisFunctionsMatrix since we are not interested in storing multiple components
        return self.backend.FunctionsList(self._space)
        
    @override
    def create_POD_container(self):
        return self.backend.ProperOrthogonalDecomposition(self._space, self._inner_product)
        
    @override
    def name(self):
        return self._name
        
    @override
    def description(self):
        return self._description
        
    @override
    def is_parametrized(self):
        return self.wrapping.is_parametrized(self._expression, self.wrapping.expression_iterator) or self.is_time_dependent()
        
    @override
    def is_time_dependent(self):
        return self.wrapping.is_time_dependent(self._expression, self.wrapping.expression_iterator)
        
class PrettyTuple(tuple):
    def __new__(cls, arg0, arg1, arg2):
        as_list = [str(arg0) + ",", "where"]
        as_list.extend([str(key) + " = " + value for key, value in arg1.iteritems()])
        as_list.append("with id " + str(arg2))
        return tuple.__new__(cls, tuple(as_list))
        
