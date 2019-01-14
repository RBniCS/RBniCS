# Copyright (C) 2015-2019 by the RBniCS authors
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
from rbnics.utils.decorators import BackendFor
from rbnics.backends.abstract import ParametrizedTensorFactory as AbstractParametrizedTensorFactory

@BackendFor("common", inputs=(Number, ))
class ParametrizedTensorFactory(AbstractParametrizedTensorFactory):
    def __init__(self, scalar):
        AbstractParametrizedTensorFactory.__init__(self, scalar)
    
    def create_interpolation_locations_container(self):
        raise RuntimeError("This method should have never been called.")
        
    def create_snapshots_container(self):
        raise RuntimeError("This method should have never been called.")
        
    def create_empty_snapshot(self):
        raise RuntimeError("This method should have never been called.")
        
    def create_basis_container(self):
        raise RuntimeError("This method should have never been called.")
        
    def create_POD_container(self):
        raise RuntimeError("This method should have never been called.")
        
    def name(self):
        raise RuntimeError("This method should have never been called.")
        
    def description(self):
        raise RuntimeError("This method should have never been called.")
        
    def is_parametrized(self):
        return False
        
    def is_time_dependent(self):
        return False
