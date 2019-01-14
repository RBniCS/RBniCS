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

def Function(Vector):
    class _Function_Type(object):
        def __init__(self, arg):
            assert isinstance(arg, (int, dict, Vector.Type()))
            if isinstance(arg, (int, dict)):
                self._v = Vector(arg)
            elif isinstance(arg, Vector.Type()):
                self._v = arg
            else: # impossible to arrive here anyway, thanks to the assert
                raise TypeError("Invalid arguments in Function")
        
        def vector(self):
            return self._v
            
        @property
        def N(self):
            return self._v.N
            
        def __abs__(self):
            v_abs = self._v.__abs__()
            function_abs = _Function_Type.__new__(type(self), v_abs)
            function_abs.__init__(v_abs)
            return function_abs
            
        def __neg__(self):
            v_neg = self._v.neg()
            function_neg = _Function_Type.__new__(type(self), v_neg)
            function_neg.__init__(v_neg)
            return function_neg
            
        def __add__(self, other):
            if isinstance(other, _Function_Type):
                v_sum = self._v.__add__(other._v)
                function_sum = _Function_Type.__new__(type(self), v_sum)
                function_sum.__init__(v_sum)
                return function_sum
            elif isinstance(other, Vector.Type()):
                v_sum = self._v.__add__(other)
                function_sum = _Function_Type.__new__(type(self), v_sum)
                function_sum.__init__(v_sum)
                return function_sum
            else:
                return NotImplemented
                
        def __iadd__(self, other):
            if isinstance(other, _Function_Type):
                self._v.__iadd__(other._v)
                return self
            elif isinstance(other, Vector.Type()):
                self._v.__iadd__(other)
                return self
            else:
                return NotImplemented
            
        def __sub__(self, other):
            if isinstance(other, _Function_Type):
                v_sub = self._v.__sub__(other._v)
                function_sub = _Function_Type.__new__(type(self), v_sub)
                function_sub.__init__(v_sub)
                return function_sub
            elif isinstance(other, Vector.Type()):
                v_sub = self._v.__sub__(other)
                function_sub = _Function_Type.__new__(type(self), v_sub)
                function_sub.__init__(v_sub)
                return function_sub
            else:
                return NotImplemented
                
        def __isub__(self, other):
            if isinstance(other, _Function_Type):
                self._v.__isub__(other._v)
                return self
            elif isinstance(other, Vector.Type()):
                self._v.__isub__(other)
                return self
            else:
                return NotImplemented
            
        def __mul__(self, other):
            if isinstance(other, Number):
                v_mul = self._v.__mul__(other)
                function_mul = _Function_Type.__new__(type(self), v_mul)
                function_mul.__init__(v_mul)
                return function_mul
            else:
                return NotImplemented
            
        def __rmul__(self, other):
            if isinstance(other, Number):
                v_rmul = self._v.__rmul__(other)
                function_rmul = _Function_Type.__new__(type(self), v_rmul)
                function_rmul.__init__(v_rmul)
                return function_rmul
            else:
                return NotImplemented
                
        def __imul__(self, other):
            if isinstance(other, Number):
                self._v.__imul__(other)
                return self
            else:
                return NotImplemented
                
        def __truediv__(self, other):
            if isinstance(other, Number):
                v_mul = self._v.__truediv__(other)
                function_mul = _Function_Type.__new__(type(self), v_mul)
                function_mul.__init__(v_mul)
                return function_mul
            else:
                return NotImplemented
            
        def __itruediv__(self, other):
            if isinstance(other, Number):
                self._v.__itruediv__(other)
                return self
            else:
                return NotImplemented
            
        def __str__(self):
            return str(self._v)
            
        def __iter__(self):
            return self._v.__iter__()
            
    return _Function_Type
