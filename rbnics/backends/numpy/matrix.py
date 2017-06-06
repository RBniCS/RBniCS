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

from numpy import ix_ as Slicer, ndarray as SlicerInnerType, matrix as MatrixBaseType
from rbnics.backends.numpy.function import Function
from rbnics.backends.numpy.vector import Vector
from rbnics.backends.numpy.wrapping import slice_to_array, slice_to_size

class _Matrix_Type(MatrixBaseType): # inherit to make sure that matrices and vectors correspond to two different types
    def __getitem__(self, key):
        if (
            (isinstance(key, tuple) and isinstance(key[0], slice)) # direct call of matrix[:5, :5]
                or
            (isinstance(key, tuple) and isinstance(key[0], tuple)) # indirect call through AffineExpansionStorage
        ):
            assert len(key) == 2
            if isinstance(key[0], slice): # direct call of matrix[:5, :5]
                # Prepare output
                if hasattr(self, "_component_name_to_basis_component_length"):
                    output = MatrixBaseType.__getitem__(self, Slicer(*slice_to_array(self, key, self._component_name_to_basis_component_length, self._component_name_to_basis_component_index)))
                    output_size = slice_to_size(self, key, self._component_name_to_basis_component_length)
                else:
                    output = MatrixBaseType.__getitem__(self, Slicer(*slice_to_array(self, key)))
                    output_size = slice_to_size(self, key)
                # Preserve M and N
                assert len(output_size) == 2
                output.M = output_size[0]
                output.N = output_size[1]
                # Return
                return output
            elif isinstance(key[0], tuple): # indirect call through AffineExpansionStorage
                return MatrixBaseType.__getitem__(self, Slicer(*key))
                # Do not preserve M and N, it will be done in AffineExpansionStorage
        else:
            return MatrixBaseType.__getitem__(self, key)
            
    def __setitem__(self, key, value):
        if (
            (isinstance(key, tuple) and isinstance(key[0], slice)) # direct call of matrix[:5, :5]
        ):
            assert len(key) == 2
            # Convert slices
            if hasattr(self, "_component_name_to_basis_component_length"):
                converted_key = Slicer(*slice_to_array(self, key, self._component_name_to_basis_component_length, self._component_name_to_basis_component_index))
            else:
                converted_key = Slicer(*slice_to_array(self, key))
            # Set item
            MatrixBaseType.__setitem__(self, converted_key, value)
        else:
            MatrixBaseType.__setitem__(self, key, value)
            
    def __abs__(self, other):
        output = MatrixBaseType.__abs__(self)
        self._arithmetic_operations_preserve_attributes(None, output, other_order=0)
        return output
        
    def __add__(self, other):
        output = MatrixBaseType.__add__(self, other)
        self._arithmetic_operations_preserve_attributes(other, output)
        return output
        
    def __sub__(self, other):
        output = MatrixBaseType.__sub__(self, other)
        self._arithmetic_operations_preserve_attributes(other, output)
        return output
        
    def __mul__(self, other):
        if isinstance(other, (Function.Type(), Vector.Type())):
            if isinstance(other, Function.Type()):
                output_as_matrix = MatrixBaseType.__mul__(self, other.vector())
            else:
                output_as_matrix = MatrixBaseType.__mul__(self, other)
            output = Vector(self.M)
            output[:] = output_as_matrix[:]
        else:
            output = MatrixBaseType.__mul__(self, other)
        if isinstance(other, (float, int)):
            self._arithmetic_operations_preserve_attributes(other, output, other_order=0)
        elif isinstance(other, Function.Type()):
            self._arithmetic_operations_preserve_attributes(other.vector(), output, other_order=1)
        elif isinstance(other, Vector.Type()):
            self._arithmetic_operations_preserve_attributes(other, output, other_order=1)
        return output
        
    def __rmul__(self, other):
        output = MatrixBaseType.__rmul__(self, other)
        if isinstance(other, (float, int)):
            self._arithmetic_operations_preserve_attributes(other, output, other_order=0)
        return output
        
    def __neg__(self):
        output = MatrixBaseType.__neg__(self)
        self._arithmetic_operations_preserve_attributes(None, output, other_order=0)
        return output
        
    def _arithmetic_operations_preserve_attributes(self, other, output, other_order=2):
        # Preserve M and N
        if other_order in (1, 2):
            assert isinstance(self.N, (int, dict))
            if isinstance(self.N, int) and isinstance(other.N, dict):
                assert len(other.N) == 1
                assert other.N.values()[0] == self.N
            elif isinstance(self.N, dict) and isinstance(other.N, int):
                assert len(self.N) == 1
                assert self.N.values()[0] == other.N
            else:
                assert self.N == other.N
        if other_order is 2:
            assert isinstance(self.M, (int, dict))
            if isinstance(self.M, int) and isinstance(other.M, dict):
                assert len(other.M) == 1
                assert other.M.values()[0] == self.M
            elif isinstance(self.M, dict) and isinstance(other.M, int):
                assert len(self.M) == 1
                assert self.M.values()[0] == other.M
            else:
                assert self.M == other.M
        output.M = self.M
        output.N = self.N
        # Preserve auxiliary attributes related to basis functions matrix
        assert hasattr(self, "_basis_component_index_to_component_name") == hasattr(self, "_component_name_to_basis_component_index")
        assert hasattr(self, "_basis_component_index_to_component_name") == hasattr(self, "_component_name_to_basis_component_length")
        if hasattr(self, "_basis_component_index_to_component_name"):
            if other_order is 2:
                if hasattr(other, "_basis_component_index_to_component_name"):
                    assert self._basis_component_index_to_component_name == other._basis_component_index_to_component_name
                if hasattr(other, "_component_name_to_basis_component_index"):
                    assert self._component_name_to_basis_component_index == other._component_name_to_basis_component_index
                if hasattr(other, "_component_name_to_basis_component_length"):
                    assert self._component_name_to_basis_component_length == other._component_name_to_basis_component_length
            elif other_order is 1:
                if hasattr(other, "_basis_component_index_to_component_name"):
                    assert self._basis_component_index_to_component_name[1] == other._basis_component_index_to_component_name
                if hasattr(other, "_component_name_to_basis_component_index"):
                    assert self._component_name_to_basis_component_index[1] == other._component_name_to_basis_component_index
                if hasattr(other, "_component_name_to_basis_component_length"):
                    assert self._component_name_to_basis_component_length[1] == other._component_name_to_basis_component_length
            if other_order is 0 or other_order is 2:
                output._basis_component_index_to_component_name = self._basis_component_index_to_component_name
                output._component_name_to_basis_component_index = self._component_name_to_basis_component_index
                output._component_name_to_basis_component_length = self._component_name_to_basis_component_length
            elif other_order is 1:
                output._basis_component_index_to_component_name = self._basis_component_index_to_component_name[1]
                output._component_name_to_basis_component_index = self._component_name_to_basis_component_index[1]
                output._component_name_to_basis_component_length = self._component_name_to_basis_component_length[1]
    
from numpy import zeros as _MatrixContent_Base
from rbnics.utils.decorators import backend_for, OnlineSizeType

@backend_for("numpy", inputs=(OnlineSizeType, OnlineSizeType), output=_Matrix_Type)
def Matrix(M, N):
    assert isinstance(M, (int, dict))
    assert isinstance(N, (int, dict))
    assert isinstance(M, dict) == isinstance(N, dict)
    if isinstance(M, dict):
        M_sum = sum(M.values())
        N_sum = sum(N.values())
    else:
        M_sum = M
        N_sum = N
    output = _Matrix_Type(_MatrixContent_Base((M_sum, N_sum)))
    output.M = M
    output.N = N
    return output
    
