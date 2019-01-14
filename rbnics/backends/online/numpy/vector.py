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

from numpy import zeros
from rbnics.backends.online.basic import Vector as BasicVector
from rbnics.backends.online.numpy.wrapping import Slicer
from rbnics.utils.decorators import backend_for, ModuleWrapper, OnlineSizeType

def VectorBaseType(N):
    return zeros(N)

backend = ModuleWrapper()
wrapping = ModuleWrapper(Slicer=Slicer)
_Vector_Type_Base = BasicVector(backend, wrapping, VectorBaseType)

class _Vector_Type(_Vector_Type_Base):
    def __getitem__(self, key):
        if isinstance(key, int):
            return float(_Vector_Type_Base.__getitem__(self, key)) # convert from numpy numbers wrappers
        else:
            return _Vector_Type_Base.__getitem__(self, key)
            
    def __iter__(self):
        return map(float, self.content.flat)
        
    def __array__(self, dtype=None):
        return self.content.__array__(dtype)

@backend_for("numpy", inputs=(OnlineSizeType, ))
def Vector(N):
    return _Vector_Type(N)
    
# Attach a Type() function
def Type():
    return _Vector_Type
Vector.Type = Type
