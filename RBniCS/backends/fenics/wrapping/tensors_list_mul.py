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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends.fenics.wrapping.tensor_copy import tensor_copy
from RBniCS.backends.numpy.function import Function  as OnlineFunction
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector

def tensors_list_mul_online_function(tensors_list, online_function):
    assert isinstance(online_function, OnlineFunction.Type())
    online_vector = online_function.vector()
    
    output = tensor_copy(tensors_list._list[0])
    output.zero()
    assert isinstance(output, (Matrix.Type(), Vector.Type()))
    if isinstance(output, Matrix.Type()):
        for (i, matrix_i) in enumerate(tensors_list._list):
            output += matrix_i*online_vector.item(i)
    elif isinstance(output, Vector.Type()):
        for (i, vector_i) in enumerate(tensors_list._list):
            output.add_local(vector_i.array()*online_vector.item(i))
        output.apply("add")
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in tensors_list_mul_online_function.")
    return output
    
