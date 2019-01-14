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

from rbnics.utils.decorators import overload

def basic_tensor_copy(backend, wrapping):
    @overload
    def _basic_tensor_copy(tensor: backend.Matrix.Type()):
        m = backend.Matrix(tensor.M, tensor.N)
        m[:, :] = tensor
        # Preserve auxiliary attributes related to basis functions matrix
        m._component_name_to_basis_component_index = tensor._component_name_to_basis_component_index
        m._component_name_to_basis_component_length = tensor._component_name_to_basis_component_length
        # Return
        return m
        
    @overload
    def _basic_tensor_copy(tensor: backend.Vector.Type()):
        v = backend.Vector(tensor.N)
        v[:] = tensor
        # Preserve auxiliary attributes related to basis functions matrix
        v._component_name_to_basis_component_index = tensor._component_name_to_basis_component_index
        v._component_name_to_basis_component_length = tensor._component_name_to_basis_component_length
        # Return
        return v
        
    return _basic_tensor_copy
