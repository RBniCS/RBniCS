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

def Function(Vector):
    class _Function_Type(object):
        def __init__(self, arg):
            assert isinstance(arg, (int, dict, Vector.Type()))
            if isinstance(arg, (int, dict)):
                self._v = Vector(arg)
                self.N = arg
            elif isinstance(arg, Vector.Type()):
                self._v = arg
                self.N = arg.N
            else: # impossible to arrive here anyway, thanks to the assert
                raise AssertionError("Invalid arguments in Function")
        
        def vector(self):
            return self._v
            
        def __abs__(self):
            return _Function_Type(self._v.__abs__())
            
        def __add__(self, other):
            if isinstance(other, _Function_Type):
                return _Function_Type(self._v.__add__(other._v))
            elif isinstance(other, Vector.Type()):
                return _Function_Type(self._v.__add__(other))
            else:
                return NotImplemented
            
        def __sub__(self, other):
            if isinstance(other, _Function_Type):
                return _Function_Type(self._v.__sub__(other._v))
            elif isinstance(other, Vector.Type()):
                return _Function_Type(self._v.__sub__(other))
            else:
                return NotImplemented
            
        def __mul__(self, other):
            if isinstance(other, (float, int)):
                return _Function_Type(self._v.__mul__(other))
            else:
                return NotImplemented
            
        def __rmul__(self, other):
            if isinstance(other, (float, int)):
                return _Function_Type(self._v.__rmul__(other))
            else:
                return NotImplemented
            
        def __neg__(self):
            return _Function_Type(self._v.neg())
            
    return _Function_Type
