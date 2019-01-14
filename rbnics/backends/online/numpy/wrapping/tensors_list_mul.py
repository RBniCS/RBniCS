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

from rbnics.backends.abstract import TensorsList as AbstractTensorsList # used in place of concrete TensorsList to avoid unsolvable circular dependency
from rbnics.utils.decorators import overload

def basic_tensors_list_mul_online_function(backend, wrapping):
    def _basic_tensors_list_mul_online_function(tensors_list, online_function):
        output = wrapping.tensor_copy(tensors_list._list[0])
        _multiply(tensors_list, online_function, output)
        return output
        
    @overload
    def _multiply(tensors_list: AbstractTensorsList, online_function: backend.Function.Type(), output: backend.Matrix.Type()):
        output[:, :] = 0.
        for (i, tensor_i) in enumerate(tensors_list._list):
            online_vector_i = online_function.vector()[i]
            output[:, :] += tensor_i*online_vector_i
            
    @overload
    def _multiply(tensors_list: AbstractTensorsList, online_function: backend.Function.Type(), output: backend.Vector.Type()):
        output[:] = 0.
        for (i, tensor_i) in enumerate(tensors_list._list):
            online_vector_i = online_function.vector()[i]
            output[:] += tensor_i*online_vector_i
            
    return _basic_tensors_list_mul_online_function
    
# No explicit instantiation for backend = rbnics.backends.online.numpy to avoid
# circular dependencies. The concrete instatiation will be carried out in
# rbnics.backends.online.numpy.tensors_list
