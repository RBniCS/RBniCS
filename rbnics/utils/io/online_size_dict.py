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

from functools import total_ordering
from collections import OrderedDict
from rbnics.utils.decorators import dict_of, overload

@total_ordering
class OnlineSizeDict(OrderedDict):
    __slots__ = ()
    
    def __init__(self, *args, **kwargs):
        super(OnlineSizeDict, self).__init__(*args, **kwargs)
        
    def __getitem__(self, k):
        return super(OnlineSizeDict, self).__getitem__(k)
        
    def __setitem__(self, k, v):
        return super(OnlineSizeDict, self).__setitem__(k, v)
        
    def __delitem__(self, k):
        return super(OnlineSizeDict, self).__delitem__(k)
        
    def get(self, k, default=None):
        return super(OnlineSizeDict, self).get(k, default)
        
    def setdefault(self, k, default=None):
        return super(OnlineSizeDict, self).setdefault(k, default)
        
    def pop(self, k):
        return super(OnlineSizeDict, self).pop(k)
        
    def update(self, **kwargs):
        super(OnlineSizeDict, self).update(**kwargs)
        
    def __contains__(self, k):
        return super(OnlineSizeDict, self).__contains__(k)
        
    # Override N += N_bc so that it is possible to increment online size due to boundary conditions (several components)
    @overload(lambda cls: cls)
    def __iadd__(self, other):
        for key in self:
            self[key] += other[key]
        return self
        
    # Override N += N_bc so that it is possible to increment online size due to boundary conditions (single component)
    @overload(int)
    def __iadd__(self, other):
        assert len(self) is 1
        for key in self:
            self[key] += other
        return self
        
    # Override __eq__ so that it is possible to check equality of dictionary with an int
    @overload(int)
    def __eq__(self, other):
        for (key, value) in self.items():
            if value != other:
                return False
        return True
        
    @overload((lambda cls: cls, dict_of(str, int)))
    def __eq__(self, other):
        return super(OnlineSizeDict, self).__eq__(other)
            
    # Override __ne__ so that it is possible to check not equality of dictionary with an int
    @overload(int)
    def __ne__(self, other):
        for (key, value) in self.items():
            if value == other:
                return False
        return True
        
    @overload((lambda cls: cls, dict_of(str, int)))
    def __ne__(self, other):
        return super(OnlineSizeDict, self).__ne__(other)
            
    # Override __lt__ so that it is possible to check if dictionary is less than an int
    @overload(int)
    def __lt__(self, other):
        for (key, value) in self.items():
            if value >= other:
                return False
        return True
        
    @overload((lambda cls: cls, dict_of(str, int)))
    def __lt__(self, other):
        return super(OnlineSizeDict, self).__lt__(other)
