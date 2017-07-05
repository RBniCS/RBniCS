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

from rbnics.backends.numpy.wrapping.tensor_copy import tensor_copy
import rbnics.backends.numpy

def tensors_list_mul_online_function(tensors_list, online_function):
    assert isinstance(online_function, rbnics.backends.numpy.Function.Type())
    online_vector = online_function.vector()
    
    output = tensor_copy(tensors_list._list[0])
    output[:] = 0.
    for (i, tensor_i) in enumerate(tensors_list._list):
        output[:] += tensor_i*online_vector.item(i)
    return output
    
