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

from rbnics.backends.basic import copy as basic_copy
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping.function_copy import basic_function_copy
from rbnics.backends.online.numpy.wrapping.tensor_copy import basic_tensor_copy
from rbnics.utils.decorators import backend_for, list_of, ModuleWrapper

backend = ModuleWrapper(Function, Matrix, Vector)
wrapping_for_wrapping = ModuleWrapper()
function_copy = basic_function_copy(backend, wrapping_for_wrapping)
tensor_copy = basic_tensor_copy(backend, wrapping_for_wrapping)
wrapping = ModuleWrapper(function_copy=function_copy, tensor_copy=tensor_copy)
copy_base = basic_copy(backend, wrapping)

@backend_for("numpy", inputs=((Function.Type(), list_of(Function.Type()), Matrix.Type(), Vector.Type()), ))
def copy(arg):
    return copy_base(arg)
