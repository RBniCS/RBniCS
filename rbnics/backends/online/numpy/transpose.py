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

from rbnics.backends.basic import transpose as basic_transpose
import rbnics.backends.online.numpy
from rbnics.backends.online.numpy.basis_functions_matrix import BasisFunctionsMatrix
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.functions_list import FunctionsList
from rbnics.backends.online.numpy.vector import Vector
from rbnics.utils.decorators import backend_for

@backend_for("numpy", inputs=((BasisFunctionsMatrix, Function.Type(), FunctionsList, Vector.Type()), ))
def transpose(arg):
    return basic_transpose(arg, rbnics.backends.online.numpy, rbnics.backends.online.numpy.wrapping)
    
