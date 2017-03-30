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
## @file transpose.py
#  @brief transpose method to be used in RBniCS.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.backends.basic import transpose as basic_transpose
import rbnics.backends.numpy
from rbnics.backends.numpy.basis_functions_matrix import BasisFunctionsMatrix
from rbnics.backends.numpy.function import Function
from rbnics.backends.numpy.functions_list import FunctionsList
from rbnics.backends.numpy.vector import Vector
import rbnics.backends.numpy.wrapping
from rbnics.utils.decorators import backend_for

@backend_for("numpy", online_backend="numpy", inputs=((BasisFunctionsMatrix, Function.Type(), FunctionsList, Vector.Type()), ))
def transpose(arg):
    return basic_transpose(arg, rbnics.backends.numpy, rbnics.backends.numpy.wrapping, rbnics.backends.numpy)
    
