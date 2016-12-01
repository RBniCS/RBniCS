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

from numpy import ix_ as Slicer, matrix as VectorBaseType
from RBniCS.backends.numpy.wrapping_utils import slice_to_array, slice_to_size

class _Vector_Type(VectorBaseType): # inherit to make sure that matrices and vectors correspond to two different types
    def __getitem__(self, key):
        if (
            isinstance(key, slice)  # direct call of vector[:5]
                or
            (isinstance(key, tuple) and isinstance(key[0], tuple)) # indirect call through AffineExpansionStorage
        ):
            if isinstance(key, slice): # direct call of vector[:5]
                # Prepare output
                output = VectorBaseType.__getitem__(self, Slicer(*slice_to_array(key, self)))
                if hasattr(self, "_component_name_to_basis_component_length"):
                    output_size = slice_to_size(key, self._component_name_to_basis_component_length)
                else:
                    output_size = slice_to_size(key)
                # Preserve N
                assert len(output_size) == 1
                output.N = output_size[0]
                # Return
                return output
            elif isinstance(key, tuple): # indirect call through AffineExpansionStorage
                assert len(key) == 1
                return VectorBaseType.__getitem__(self, Slicer(*key))
                # Do not preserve N, it will be done in AffineExpansionStorage
        elif isinstance(key, int):
            output = VectorBaseType.__getitem__(self, key)
            output.N = 1
            return output
        else:
            return VectorBaseType.__getitem__(self, key)
            
    def __abs__(self):
        output = VectorBaseType.__abs__(self)
        self._arithmetic_operations_preserve_attributes(None, output, other_is_vector=False)
        return output
        
    def __add__(self, other):
        output = VectorBaseType.__add__(self, other)
        self._arithmetic_operations_preserve_attributes(other, output)
        return output
        
    def __sub__(self, other):
        output = VectorBaseType.__sub__(self, other)
        self._arithmetic_operations_preserve_attributes(other, output)
        return output
        
    def __mul__(self, other):
        output = VectorBaseType.__mul__(self, other)
        if isinstance(other, float):
            self._arithmetic_operations_preserve_attributes(other, output, other_is_vector=False)
        return output
        
    def __rmul__(self, other):
        output = VectorBaseType.__rmul__(self, other)
        if isinstance(other, float):
            self._arithmetic_operations_preserve_attributes(other, output, other_is_vector=False)
        return output
        
    def __neg__(self):
        output = VectorBaseType.__neg__(self)
        self._arithmetic_operations_preserve_attributes(None, output, other_is_vector=False)
        return output
        
    def _arithmetic_operations_preserve_attributes(self, other, output, other_is_vector=True):
        # Preserve N
        if other_is_vector:
            assert self.N == other.N
        output.N = self.N
        # Preserve auxiliary attributes related to basis functions matrix
        assert hasattr(self, "_basis_component_index_to_component_name") == hasattr(self, "_component_name_to_basis_component_index")
        assert hasattr(self, "_basis_component_index_to_component_name") == hasattr(self, "_component_name_to_basis_component_length")
        if hasattr(self, "_basis_component_index_to_component_name"):
            if other_is_vector:
                assert hasattr(other, "_basis_component_index_to_component_name")
                assert hasattr(other, "_component_name_to_basis_component_index")
                assert hasattr(other, "_component_name_to_basis_component_length")
                assert self._basis_component_index_to_component_name == other._basis_component_index_to_component_name
                assert self._component_name_to_basis_component_index == other._component_name_to_basis_component_index
                assert self._component_name_to_basis_component_length == other._component_name_to_basis_component_length
            output._basis_component_index_to_component_name = self._basis_component_index_to_component_name
            output._component_name_to_basis_component_index = self._component_name_to_basis_component_index
            output._component_name_to_basis_component_length = self._component_name_to_basis_component_length

from numpy import zeros as _VectorContent_Base
from RBniCS.utils.decorators import backend_for, OnlineSizeType

@backend_for("NumPy", inputs=(OnlineSizeType, ), output=_Vector_Type)
def Vector(N):
    assert isinstance(N, (int, dict))
    if isinstance(N, dict):
        N_sum = sum(N.values())
    else:
        N_sum = N
    output = _Vector_Type(_VectorContent_Base((N_sum))).transpose() # as column vector
    output.N = N
    return output
    
