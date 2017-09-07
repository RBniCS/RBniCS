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

from abc import ABCMeta, abstractmethod

class Distribution(object, metaclass=ABCMeta):
    @abstractmethod
    def sample(self, box, n):
        raise NotImplementedError("The method sample is distribution-specific and needs to be overridden.")
        
    ## Override the following methods to use a Distribution as a dict key
    def __hash__(self):
        dict_for_hash = list()
        for (k, v) in self.__dict__.items():
            if isinstance(v, dict):
                dict_for_hash.append( tuple(v.values()) )
            elif isinstance(v, list):
                dict_for_hash.append( tuple(v) )
            else:
                dict_for_hash.append(v)
        return hash((type(self).__name__, tuple(dict_for_hash)))
        
    def __eq__(self, other):
        return (type(self).__name__, self.__dict__) == (type(other).__name__, other.__dict__)
        
    def __ne__(self, other):
        return not(self == other)
        
