# Copyright (C) 2015-2018 by the RBniCS authors
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

def ParametrizedExpressionFactory(backend, wrapping):
    class _ParametrizedExpressionFactory(AbstractParametrizedExpressionFactory):
        def __init__(self, expression, space, inner_product):
            AbstractParametrizedExpressionFactory.__init__(self, expression)
            self._expression = expression
            self._space = space
            self._inner_product = inner_product
            self._name = wrapping.expression_name(expression)
            self._description = PrettyTuple(self._expression, wrapping.expression_description(self._expression), self._name)
            
        def __eq__(self, other):
            return (
                isinstance(other, type(self))
                    and
                self._expression == other._expression
                    and
                self._space == other._space
                    and
                self._inner_product == other._inner_product
            )
            
        def __hash__(self):
            return hash((self._name, self._space, self._inner_product))
                
        def create_interpolation_locations_container(self):
            return backend.ReducedVertices(self._space)
            
        def create_snapshots_container(self):
            return backend.SnapshotsMatrix(self._space)
            
        def create_empty_snapshot(self):
            return backend.Function(self._space)
            
        def create_basis_container(self):
            # We use FunctionsList instead of BasisFunctionsMatrix since we are not interested in storing multiple components
            return backend.FunctionsList(self._space)
            
        def create_POD_container(self):
            return backend.ProperOrthogonalDecomposition(self._space, self._inner_product)
            
        def name(self):
            return self._name
            
        def description(self):
            return self._description
            
        def is_parametrized(self):
            return wrapping.is_parametrized(self._expression, wrapping.expression_iterator) or self.is_time_dependent()
            
        def is_time_dependent(self):
            return wrapping.is_time_dependent(self._expression, wrapping.expression_iterator)
    return _ParametrizedExpressionFactory
        
class PrettyTuple(tuple):
    def __new__(cls, arg0, arg1, arg2):
        as_list = [str(arg0) + ",", "where"]
        as_list.extend([str(key) + " = " + value for key, value in arg1.items()])
        as_list.append("with id " + str(arg2))
        return tuple.__new__(cls, tuple(as_list))
