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

from numpy import matrix
from rbnics.utils.decorators import backend_for, OnlineSizeType
from rbnics.backends.online.basic import Function as BasicFunction
from rbnics.backends.online.numpy.vector import Vector

_Function_Type_Base = BasicFunction(Vector)

class _Function_Type(_Function_Type_Base):
    def __init__(self, arg):
        assert isinstance(arg, (int, dict, Vector.Type(), matrix))
        if isinstance(arg, (int, dict, Vector.Type())):
            _Function_Type_Base.__init__(self, arg)
        elif isinstance(arg, matrix): # for internal usage in EigenSolver, not exposed to the backends
            assert arg.shape[1] == 1 # column vector
            vec = Vector(arg.shape[0])
            vec[:] = arg
            _Function_Type_Base.__init__(self, vec)
        else: # impossible to arrive here anyway, thanks to the assert
            raise TypeError("Invalid arguments in Function")
                    
    def __iter__(self):
        return map(float, self._v.flat)
        
@backend_for("numpy", inputs=(OnlineSizeType + (Vector.Type(), ), ))
def Function(arg):
    return _Function_Type(arg)
    
# Attach a Type() function
def Type():
    return _Function_Type
Function.Type = Type
