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
## @file online_vector.py
#  @brief Type of online vector
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import GenericMatrix
from RBniCS.backends.fenics.function import Function

def Matrix():
    raise NotImplementedError("This is dummy function (not required by the interface) just store the Type")
    
def _Matrix_Type():
    return GenericMatrix
Matrix.Type = _Matrix_Type

# Enable matrix*function product (i.e. matrix*function.vector())
original__mul__ = GenericMatrix.__mul__
def custom__mul__(self, other):
    if isinstance(other, Function.Type()):
        return original__mul__(self, other.vector())
    else:
        return original__mul__(self, other)
GenericMatrix.__mul__ = custom__mul__

