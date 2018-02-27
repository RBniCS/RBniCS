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

from rbnics.backends.abstract import ParametrizedTensorFactory as AbstractParametrizedTensorFactory

def ParametrizedTensorFactory(backend, wrapping):
    class _ParametrizedTensorFactory(AbstractParametrizedTensorFactory):
        def __init__(self, form, spaces, empty_snapshot):
            AbstractParametrizedTensorFactory.__init__(self, form)
            self._form = form
            self._name = wrapping.form_name(form)
            self._description = PrettyTuple(self._form, wrapping.form_description(self._form), self._name)
            self._spaces = spaces
            self._empty_snapshot = empty_snapshot
            
        def __eq__(self, other):
            return (
                isinstance(other, type(self))
                    and
                self._name == other._name
                    and
                self._spaces == other._spaces
            )
            
        def __hash__(self):
            return hash((self._name, self._spaces))
        
        def create_interpolation_locations_container(self, **kwargs):
            return backend.ReducedMesh(self._spaces, **kwargs)
            
        def create_snapshots_container(self):
            return backend.TensorSnapshotsList(self._spaces, self._empty_snapshot)
            
        def create_empty_snapshot(self):
            return backend.copy(self._empty_snapshot)
            
        def create_basis_container(self):
            return backend.TensorBasisList(self._spaces, self._empty_snapshot)
            
        def create_POD_container(self):
            return backend.HighOrderProperOrthogonalDecomposition(self._spaces, self._empty_snapshot)
            
        def name(self):
            return self._name
            
        def description(self):
            return self._description
            
        def is_parametrized(self):
            return wrapping.is_parametrized(self._form, wrapping.form_iterator) or self.is_time_dependent()
            
        def is_time_dependent(self):
            return wrapping.is_time_dependent(self._form, wrapping.form_iterator)
    return _ParametrizedTensorFactory
        
class PrettyTuple(tuple):
    def __new__(cls, arg0, arg1, arg2):
        as_list = [str(arg0) + ",", "where"]
        as_list.extend([str(key) + " = " + value for key, value in arg1.items()])
        as_list.append("with id " + str(arg2))
        return tuple.__new__(cls, tuple(as_list))
