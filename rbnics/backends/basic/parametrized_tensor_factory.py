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

from rbnics.backends.abstract import ParametrizedTensorFactory as AbstractParametrizedTensorFactory
from rbnics.utils.decorators import Extends, override

@Extends(AbstractParametrizedTensorFactory)
class ParametrizedTensorFactory(AbstractParametrizedTensorFactory):
    def __init__(self, form, spaces, empty_snapshot, backend, wrapping):
        AbstractParametrizedTensorFactory.__init__(self, form)
        self._form = form
        self._name = wrapping.form_name(form)
        self._description = PrettyTuple(self._form, wrapping.form_description(self._form), self._name)
        self._spaces = spaces
        self._empty_snapshot = empty_snapshot
        self.backend = backend
        self.wrapping = wrapping
    
    @override
    def create_interpolation_locations_container(self, **kwargs):
        return self.backend.ReducedMesh(self._spaces, **kwargs)
        
    @override
    def create_snapshots_container(self):
        return self.backend.TensorSnapshotsList(self._spaces, self._empty_snapshot)
        
    @override
    def create_empty_snapshot(self):
        return self.backend.copy(self._empty_snapshot)
        
    @override
    def create_basis_container(self):
        return self.backend.TensorBasisList(self._spaces, self._empty_snapshot)
        
    @override
    def create_POD_container(self):
        return self.backend.HighOrderProperOrthogonalDecomposition(self._spaces, self._empty_snapshot)
        
    @override
    def name(self):
        return self._name
        
    @override
    def description(self):
        return self._description
        
    @override
    def is_parametrized(self):
        return self.wrapping.is_parametrized(self._form, self.wrapping.form_iterator) or self.is_time_dependent()
        
    @override
    def is_time_dependent(self):
        return self.wrapping.is_time_dependent(self._form, self.wrapping.form_iterator)
        
class PrettyTuple(tuple):
    def __new__(cls, arg0, arg1, arg2):
        as_list = [str(arg0) + ",", "where"]
        as_list.extend([str(key) + " = " + value for key, value in arg1.items()])
        as_list.append("with id " + str(arg2))
        return tuple.__new__(cls, tuple(as_list))
        
