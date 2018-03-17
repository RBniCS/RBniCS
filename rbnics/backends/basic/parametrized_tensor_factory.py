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

from numbers import Number
from rbnics.backends.abstract import ParametrizedTensorFactory as AbstractParametrizedTensorFactory
from rbnics.eim.utils.decorators import add_to_map_from_parametrized_operator_to_problem, get_problem_from_parametrized_operator
from rbnics.utils.decorators import overload

def ParametrizedTensorFactory(backend, wrapping):
    class _ParametrizedTensorFactory(AbstractParametrizedTensorFactory):
        def __init__(self, form, spaces, assemble_empty_snapshot):
            AbstractParametrizedTensorFactory.__init__(self, form)
            self._form = form
            self._spaces = spaces
            self._assemble_empty_snapshot = assemble_empty_snapshot
            self._empty_snapshot = None
            self._name = None
            self._description = None
            
        def __eq__(self, other):
            return (
                isinstance(other, type(self))
                    and
                self._form == other._form
                    and
                self._spaces == other._spaces
            )
            
        def __hash__(self):
            return hash((self._form, self._spaces))
        
        def create_interpolation_locations_container(self, **kwargs):
            return backend.ReducedMesh(self._spaces, **kwargs)
            
        def create_snapshots_container(self):
            return backend.TensorSnapshotsList(self._spaces, self.create_empty_snapshot())
            
        def create_empty_snapshot(self):
            if self._empty_snapshot is None:
                self._empty_snapshot = self._assemble_empty_snapshot()
            return backend.copy(self._empty_snapshot)
            
        def create_basis_container(self):
            return backend.TensorBasisList(self._spaces, self.create_empty_snapshot())
            
        def create_POD_container(self):
            return backend.HighOrderProperOrthogonalDecomposition(self._spaces, self.create_empty_snapshot())
            
        def name(self):
            if self._name is None:
                self._name = wrapping.form_name(self._form)
            return self._name
            
        def description(self):
            if self._description is None:
                self._description = PrettyTuple(self._form, wrapping.form_description(self._form), self.name())
            return self._description
            
        def is_parametrized(self):
            return wrapping.is_parametrized(self._form, wrapping.form_iterator) or self.is_time_dependent()
            
        def is_time_dependent(self):
            return wrapping.is_time_dependent(self._form, wrapping.form_iterator)
            
        @overload(lambda cls: cls)
        def __add__(self, other):
            form_sum = self._form + other._form
            output = _ParametrizedTensorFactory.__new__(type(self), form_sum)
            output.__init__(form_sum)
            problems = [get_problem_from_parametrized_operator(operator) for operator in (self, other)]
            assert all([problem is problems[0] for problem in problems])
            add_to_map_from_parametrized_operator_to_problem(output, problems[0])
            return output
            
        @overload(lambda cls: cls)
        def __sub__(self, other):
            return self + (- other)
        
        @overload(backend.Function.Type())
        def __mul__(self, other):
            form_mul = self._form*other
            output = _ParametrizedTensorFactory.__new__(type(self), form_mul)
            output.__init__(form_mul)
            add_to_map_from_parametrized_operator_to_problem(output, get_problem_from_parametrized_operator(self))
            return output
            
        @overload(Number)
        def __rmul__(self, other):
            form_mul = other*self._form
            output = _ParametrizedTensorFactory.__new__(type(self), form_mul)
            output.__init__(form_mul)
            add_to_map_from_parametrized_operator_to_problem(output, get_problem_from_parametrized_operator(self))
            return output
            
        def __neg__(self):
            return -1.*self
        
    return _ParametrizedTensorFactory
        
class PrettyTuple(tuple):
    def __new__(cls, arg0, arg1, arg2):
        as_list = [str(arg0) + ",", "where"]
        as_list.extend([str(key) + " = " + value for key, value in arg1.items()])
        as_list.append("with id " + str(arg2))
        return tuple.__new__(cls, tuple(as_list))
