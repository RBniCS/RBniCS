# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file online_function.py
#  @brief Type of online function
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import matrix
from RBniCS.utils.decorators import backend_for
from RBniCS.backends.numpy.vector import Vector

class _Function_Type(object):
    def __init__(self, arg):
        assert isinstance(arg, (int, Vector.Type(), matrix))
        if isinstance(arg, int):
            self._v = Vector(arg)
        elif isinstance(arg, Vector.Type()):
            self._v = arg
        elif isinstance(arg, matrix): # for internal usage in EigenSolver, not exposed to the backends
            assert arg.shape[1] == 1 # column vector
            self._v = Vector(arg.shape[0])
            self._v[:] = arg
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in Function")
    
    def vector(self):
        return self._v

        
@backend_for("NumPy", inputs=((int, Vector.Type()), ), output=_Function_Type)
def Function(arg):
    return _Function_Type(arg)
    
