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
from ufl import Form
from dolfin import Constant

original__mul__ = Form.__mul__
def custom__mul__(self, other):
    if isinstance(other, Number):
        return self.__rmul__(Constant(other))
    else:
        return original__mul__(self, other)
setattr(Form, "__mul__", custom__mul__)

original__rmul__ = Form.__rmul__
def custom__rmul__(self, other):
    if isinstance(other, Number):
        return original__rmul__(self, Constant(other))
    else:
        return original__rmul__(self, other)
setattr(Form, "__rmul__", custom__rmul__)

def custom__truediv__(self, other):
    if isinstance(other, Number):
        return self.__rmul__(Constant(1./other))
    else:
        return NotImplemented
setattr(Form, "__truediv__", custom__truediv__)
