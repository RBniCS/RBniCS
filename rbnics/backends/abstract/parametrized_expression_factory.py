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

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod

@AbstractBackend
class ParametrizedExpressionFactory(object, metaclass=ABCMeta):
    def __init__(self, expression):
        pass
    
    @abstractmethod
    def create_interpolation_locations_container(self):
        pass
        
    @abstractmethod
    def create_snapshots_container(self):
        pass
        
    @abstractmethod
    def create_empty_snapshot(self):
        pass
        
    @abstractmethod
    def create_basis_container(self):
        pass
        
    @abstractmethod
    def create_POD_container(self):
        pass
        
    def interpolation_method_name(self):
        return "EIM"
        
    @abstractmethod
    def name(self):
        pass
        
    @abstractmethod
    def description(self):
        pass
    
    @abstractmethod
    def is_parametrized(self):
        pass
        
    @abstractmethod
    def is_time_dependent(self):
        pass
