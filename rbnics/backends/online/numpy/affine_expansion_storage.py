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

from rbnics.backends.online.basic import AffineExpansionStorage as BasicAffineExpansionStorage
from rbnics.backends.online.numpy.copy import function_copy, tensor_copy
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import function_load, function_save, tensor_load, tensor_save
from rbnics.utils.decorators import BackendFor, ModuleWrapper, tuple_of

backend = ModuleWrapper(Function, Matrix, Vector)
wrapping = ModuleWrapper(function_load, function_save, tensor_load, tensor_save, function_copy=function_copy, tensor_copy=tensor_copy)
AffineExpansionStorage_Base = BasicAffineExpansionStorage(backend, wrapping)

@BackendFor("numpy", inputs=((int, tuple_of(Matrix.Type()), tuple_of(Vector.Type())), (int, None)))
class AffineExpansionStorage(AffineExpansionStorage_Base):
    def __init__(self, arg1, arg2=None):
        AffineExpansionStorage_Base.__init__(self, arg1, arg2)
