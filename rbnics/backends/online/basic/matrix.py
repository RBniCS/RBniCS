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

from numbers import Number
from rbnics.backends.online.basic.wrapping import slice_to_array, slice_to_size

def Matrix(backend, wrapping, MatrixBaseType):
    class _Matrix_Type(MatrixBaseType):
        @staticmethod
        def convert_matrix_sizes_from_dicts(M, N):
            assert isinstance(M, (int, dict))
            assert isinstance(N, (int, dict))
            assert isinstance(M, dict) == isinstance(N, dict)
            if isinstance(M, dict):
                M_sum = sum(M.values())
                N_sum = sum(N.values())
            else:
                M_sum = M
                N_sum = N
            return (M_sum, N_sum)
            
        def __getitem__(self, key):
            assert isinstance(key, tuple)
            assert len(key) == 2
            key_is_tuple_of_slices = all([isinstance(key_i, slice) for key_i in key])
            key_is_tuple_of_tuples_or_lists = all([isinstance(key_i, (list, tuple)) for key_i in key])
            if (
                key_is_tuple_of_slices # direct call of matrix[:5, :5]
                    or
                key_is_tuple_of_tuples_or_lists # indirect call through AffineExpansionStorage
            ):
                assert key_is_tuple_of_slices is not key_is_tuple_of_tuples_or_lists
                if key_is_tuple_of_slices: # direct call of matrix[:5, :5]
                    # Prepare output
                    if hasattr(self, "_component_name_to_basis_component_length"):
                        output = MatrixBaseType.__getitem__(self, wrapping.Slicer(*slice_to_array(self, key, self._component_name_to_basis_component_length, self._component_name_to_basis_component_index)))
                        output_size = slice_to_size(self, key, self._component_name_to_basis_component_length)
                    else:
                        output = MatrixBaseType.__getitem__(self, wrapping.Slicer(*slice_to_array(self, key)))
                        output_size = slice_to_size(self, key)
                elif key_is_tuple_of_tuples_or_lists: # indirect call through AffineExpansionStorage
                    output = MatrixBaseType.__getitem__(self, wrapping.Slicer(*key))
                    output_size = (len(key[0]), len(key[1]))
                # Preserve M and N
                assert len(output_size) == 2
                output.M = output_size[0]
                output.N = output_size[1]
                # Preserve auxiliary attributes related to basis functions matrix
                assert hasattr(self, "_basis_component_index_to_component_name") == hasattr(self, "_component_name_to_basis_component_index")
                assert hasattr(self, "_basis_component_index_to_component_name") == hasattr(self, "_component_name_to_basis_component_length")
                if hasattr(self, "_basis_component_index_to_component_name"):
                    output._basis_component_index_to_component_name = self._basis_component_index_to_component_name
                    output._component_name_to_basis_component_index = self._component_name_to_basis_component_index
                    output._component_name_to_basis_component_length = self._component_name_to_basis_component_length
                return output
            else:
                return MatrixBaseType.__getitem__(self, key)
                
        def __setitem__(self, key, value):
            assert isinstance(key, tuple)
            assert len(key) == 2
            if all([isinstance(key_i, slice) for key_i in key]): # direct call of matrix[:5, :5]
                # Convert slices
                if hasattr(self, "_component_name_to_basis_component_length"):
                    converted_key = wrapping.Slicer(*slice_to_array(self, key, self._component_name_to_basis_component_length, self._component_name_to_basis_component_index))
                else:
                    converted_key = wrapping.Slicer(*slice_to_array(self, key))
                # Set item
                MatrixBaseType.__setitem__(self, converted_key, value)
            else:
                MatrixBaseType.__setitem__(self, key, value)
                
        def __abs__(self):
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
            if isinstance(other, (backend.Function.Type(), backend.Vector.Type(), Number)):
                if isinstance(other, (backend.Function.Type(), backend.Vector.Type())):
                    if isinstance(other, backend.Function.Type()):
                        output = MatrixBaseType.__mul__(self, other.vector())
                    else:
                        output = MatrixBaseType.__mul__(self, other)
                else:
                    output = MatrixBaseType.__mul__(self, other)
                if isinstance(other, Number):
                    self._arithmetic_operations_preserve_attributes(other, output, other_order=0)
                elif isinstance(other, backend.Function.Type()):
                    output = backend.Function(output)
                    self._arithmetic_operations_preserve_attributes(other.vector(), output, other_order=1)
                elif isinstance(other, backend.Vector.Type()):
                    self._arithmetic_operations_preserve_attributes(other, output, other_order=1)
                return output
            else:
                return NotImplemented
            
        def __rmul__(self, other):
            if isinstance(other, Number):
                output = MatrixBaseType.__rmul__(self, other)
                self._arithmetic_operations_preserve_attributes(other, output, other_order=0)
                return output
            else:
                return NotImplemented
            
        def __neg__(self):
            output = MatrixBaseType.__neg__(self)
            self._arithmetic_operations_preserve_attributes(None, output, other_order=0)
            return output
            
        def _arithmetic_operations_preserve_attributes(self, other, output, other_order=2):
            # Preserve M and N
            if other_order is 0:
                output.N = self.N
                output.M = self.M
            if other_order in (1, 2):
                assert isinstance(self.N, (int, dict))
                if isinstance(self.N, int) and isinstance(other.N, dict):
                    assert len(other.N) == 1
                    for (_, other_N) in other.N.items():
                        break
                    assert other_N == self.N
                elif isinstance(self.N, dict) and isinstance(other.N, int):
                    assert len(self.N) == 1
                    for (_, self_N) in self.N.items():
                        break
                    assert self_N == other.N
                else:
                    assert self.N == other.N
                output.N = self.N
            if other_order is 2:
                assert isinstance(self.M, (int, dict))
                if isinstance(self.M, int) and isinstance(other.M, dict):
                    assert len(other.M) == 1
                    for (_, other_M) in other.M.items():
                        break
                    assert other_M == self.M
                elif isinstance(self.M, dict) and isinstance(other.M, int):
                    assert len(self.M) == 1
                    for (_, self_M) in self.M.items():
                        break
                    assert self_M == other.M
                else:
                    assert self.M == other.M
                output.M = self.M
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
                        assert self._basis_component_index_to_component_name[0] == other._basis_component_index_to_component_name
                    if hasattr(other, "_component_name_to_basis_component_index"):
                        assert self._component_name_to_basis_component_index[0] == other._component_name_to_basis_component_index
                    if hasattr(other, "_component_name_to_basis_component_length"):
                        assert self._component_name_to_basis_component_length[0] == other._component_name_to_basis_component_length
                if other_order is 0 or other_order is 2:
                    output._basis_component_index_to_component_name = self._basis_component_index_to_component_name
                    output._component_name_to_basis_component_index = self._component_name_to_basis_component_index
                    output._component_name_to_basis_component_length = self._component_name_to_basis_component_length
                elif other_order is 1:
                    output._basis_component_index_to_component_name = self._basis_component_index_to_component_name[0]
                    output._component_name_to_basis_component_index = self._component_name_to_basis_component_index[0]
                    output._component_name_to_basis_component_length = self._component_name_to_basis_component_length[0]
    
    return _Matrix_Type
