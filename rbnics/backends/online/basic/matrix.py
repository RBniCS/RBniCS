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

from numbers import Number
from rbnics.backends.online.basic.wrapping import slice_to_array, slice_to_size
from rbnics.utils.io import ComponentNameToBasisComponentIndexDict, OnlineSizeDict

def Matrix(backend, wrapping, MatrixBaseType):
    class Matrix_Class(object):
        def __init__(self, M, N, content=None):
            assert isinstance(M, (int, dict))
            assert isinstance(N, (int, dict))
            assert isinstance(M, dict) == isinstance(N, dict)
            if isinstance(M, dict):
                M_sum = sum(M.values())
                N_sum = sum(N.values())
            else:
                M_sum = M
                N_sum = N
            self.M = M
            self.N = N
            if content is None:
                self.content = MatrixBaseType(M_sum, N_sum)
            else:
                self.content = content
            # Auxiliary attributes related to basis functions matrix
            if isinstance(M, dict):
                if len(M) > 1:
                    # ordering (stored by OnlineSizeDict, which inherits from OrderedDict) is important in the definition of attributes
                    assert isinstance(M, OnlineSizeDict)
                else:
                    self.M = M = OnlineSizeDict(M)
                if len(N) > 1:
                    # ordering (stored by OnlineSizeDict, which inherits from OrderedDict) is important in the definition of attributes
                    assert isinstance(N, OnlineSizeDict)
                else:
                    self.N = N = OnlineSizeDict(N)
                component_name_to_basis_component_index_0 = ComponentNameToBasisComponentIndexDict()
                component_name_to_basis_component_length_0 = OnlineSizeDict()
                for (component_index, (component_name, component_length)) in enumerate(M.items()):
                    component_name_to_basis_component_index_0[component_name] = component_index
                    component_name_to_basis_component_length_0[component_name] = component_length
                component_name_to_basis_component_index_1 = ComponentNameToBasisComponentIndexDict()
                component_name_to_basis_component_length_1 = OnlineSizeDict()
                for (component_index, (component_name, component_length)) in enumerate(N.items()):
                    component_name_to_basis_component_index_1[component_name] = component_index
                    component_name_to_basis_component_length_1[component_name] = component_length
                self._component_name_to_basis_component_index = (
                    component_name_to_basis_component_index_0,
                    component_name_to_basis_component_index_1
                )
                self._component_name_to_basis_component_length = (
                    component_name_to_basis_component_length_0,
                    component_name_to_basis_component_length_1
                )
            else:
                self._component_name_to_basis_component_index = (None, None)
                self._component_name_to_basis_component_length = (None, None)
            
        def __getitem__(self, key):
            assert isinstance(key, tuple)
            assert len(key) == 2
            key_is_tuple_of_slices = all([isinstance(key_i, slice) for key_i in key])
            key_is_tuple_of_tuples_or_lists = all([isinstance(key_i, (list, tuple)) for key_i in key])
            key_is_tuple_of_int = all([isinstance(key_i, int) for key_i in key])
            if (
                key_is_tuple_of_slices # matrix[:5, :5]
                    or
                key_is_tuple_of_tuples_or_lists # matrix[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
            ):
                assert key_is_tuple_of_slices is not key_is_tuple_of_tuples_or_lists
                if key_is_tuple_of_slices: # matrix[:5, :5]
                    output_content = self.content[wrapping.Slicer(*slice_to_array(self, key, self._component_name_to_basis_component_length, self._component_name_to_basis_component_index))]
                    output_size = slice_to_size(self, key, self._component_name_to_basis_component_length)
                elif key_is_tuple_of_tuples_or_lists: # matrix[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
                    output_content = self.content[wrapping.Slicer(*key)]
                    output_size = (len(key[0]), len(key[1]))
                # Prepare output
                assert len(output_size) == 2
                output = Matrix_Class.__new__(type(self), output_size[0], output_size[1], output_content)
                output.__init__(output_size[0], output_size[1], output_content)
                # Preserve auxiliary attributes related to basis functions matrix
                output._component_name_to_basis_component_index = self._component_name_to_basis_component_index
                if (
                    self._component_name_to_basis_component_length[0] is None
                        and
                    self._component_name_to_basis_component_length[1] is None
                ):
                    output._component_name_to_basis_component_length = (None, None)
                else:
                    if key_is_tuple_of_slices: # matrix[:5, :5]
                        output._component_name_to_basis_component_length = tuple(output_size)
                    elif key_is_tuple_of_tuples_or_lists: # matrix[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
                        component_name_to_basis_component_length = [None, None]
                        for i in range(2):
                            if len(self._component_name_to_basis_component_length[i]) == 1:
                                for (component_name, _) in self._component_name_to_basis_component_length[i].items():
                                    break
                                component_name_to_basis_component_length_i = OnlineSizeDict()
                                component_name_to_basis_component_length_i[component_name] = len(key[i])
                                component_name_to_basis_component_length[i] = component_name_to_basis_component_length_i
                            else:
                                raise NotImplementedError("Matrix.__getitem__ with list or tuple input arguments has not been implemented yet for the case of multiple components")
                        output._component_name_to_basis_component_length = tuple(component_name_to_basis_component_length)
                return output
            elif key_is_tuple_of_int: # matrix[5, 5]
                output = self.content[key]
                assert isinstance(output, Number)
                return output
            else:
                raise TypeError("Unsupported key type in Matrix.__getitem__")
                
        def __setitem__(self, key, value):
            assert isinstance(key, tuple)
            assert len(key) == 2
            key_is_tuple_of_slices = all([isinstance(key_i, slice) for key_i in key])
            key_is_tuple_of_slice_and_int = isinstance(key[0], slice) and isinstance(key[1], int)
            key_is_tuple_of_int_and_slice = isinstance(key[0], int) and isinstance(key[1], slice)
            key_is_tuple_of_int = all([isinstance(key_i, int) for key_i in key])
            if key_is_tuple_of_slices: # matrix[:5, :5]
                converted_key = wrapping.Slicer(*slice_to_array(self, key, self._component_name_to_basis_component_length, self._component_name_to_basis_component_index))
                if isinstance(value, type(self)):
                    value = value.content
                self.content[converted_key] = value
            elif key_is_tuple_of_slice_and_int: # matrix[:5, 5]
                converted_key_0 = wrapping.Slicer(slice_to_array(self, key[0], self._component_name_to_basis_component_length[0], self._component_name_to_basis_component_index[0]))
                converted_key = (converted_key_0, key[1])
                if isinstance(value, backend.Vector.Type()):
                    value = value.content
                self.content[converted_key] = value
            elif key_is_tuple_of_int_and_slice: # matrix[5, :5]
                converted_key_1 = wrapping.Slicer(slice_to_array(self, key[1], self._component_name_to_basis_component_length[1], self._component_name_to_basis_component_index[1]))
                converted_key = (key[0], converted_key_1)
                if isinstance(value, backend.Vector.Type()):
                    value = value.content
                self.content[converted_key] = value
            elif key_is_tuple_of_int: # matrix[5, 5]
                self.content[key] = value
            else:
                raise TypeError("Unsupported key type in Matrix.__setitem__")
                
        def __abs__(self):
            self._arithmetic_operations_assert_attributes(None, other_order=0)
            output_content = self.content.__abs__()
            output_size = (self.M, self.N)
            output = Matrix_Class.__new__(type(self), output_size[0], output_size[1], output_content)
            output.__init__(output_size[0], output_size[1], output_content)
            self._arithmetic_operations_preserve_attributes(output, other_order=0)
            return output
            
        def __neg__(self):
            self._arithmetic_operations_assert_attributes(None, other_order=0)
            output_content = self.content.__neg__()
            output_size = (self.M, self.N)
            output = Matrix_Class.__new__(type(self), output_size[0], output_size[1], output_content)
            output.__init__(output_size[0], output_size[1], output_content)
            self._arithmetic_operations_preserve_attributes(output, other_order=0)
            return output
            
        def __add__(self, other):
            if isinstance(other, type(self)):
                self._arithmetic_operations_assert_attributes(other)
                output_content = self.content.__add__(other.content)
                output_size = (self.M, self.N)
                output = Matrix_Class.__new__(type(self), output_size[0], output_size[1], output_content)
                output.__init__(output_size[0], output_size[1], output_content)
                self._arithmetic_operations_preserve_attributes(output)
                return output
            else:
                return NotImplemented
                
        def __iadd__(self, other):
            if isinstance(other, type(self)):
                self._arithmetic_operations_assert_attributes(other)
                self.content.__iadd__(other.content)
                return self
            else:
                return NotImplemented
            
        def __sub__(self, other):
            if isinstance(other, type(self)):
                self._arithmetic_operations_assert_attributes(other)
                output_content = self.content.__sub__(other.content)
                output_size = (self.M, self.N)
                output = Matrix_Class.__new__(type(self), output_size[0], output_size[1], output_content)
                output.__init__(output_size[0], output_size[1], output_content)
                self._arithmetic_operations_preserve_attributes(output)
                return output
            else:
                return NotImplemented
                
        def __isub__(self, other):
            if isinstance(other, type(self)):
                self._arithmetic_operations_assert_attributes(other)
                self.content.__isub__(other.content)
                return self
            else:
                return NotImplemented
            
        def __mul__(self, other):
            if isinstance(other, Number):
                self._arithmetic_operations_assert_attributes(other, other_order=0)
                output_content = self.content.__mul__(other)
                output_size = (self.M, self.N)
                output = Matrix_Class.__new__(type(self), output_size[0], output_size[1], output_content)
                output.__init__(output_size[0], output_size[1], output_content)
                self._arithmetic_operations_preserve_attributes(output, other_order=0)
                return output
            elif isinstance(other, backend.Vector.Type()):
                self._arithmetic_operations_assert_attributes(other, other_order=1)
                output_content = self.content.__mul__(other.content)
                output_size = self.M
                output = backend.Vector.Type()(output_size, output_content)
                self._arithmetic_operations_preserve_attributes(output, other_order=1)
                return output
            elif isinstance(other, backend.Function.Type()):
                return self.__mul__(other.vector())
            else:
                return NotImplemented
            
        def __rmul__(self, other):
            if isinstance(other, Number):
                self._arithmetic_operations_assert_attributes(other, other_order=0)
                output_content = self.content.__rmul__(other)
                output_size = (self.M, self.N)
                output = Matrix_Class.__new__(type(self), output_size[0], output_size[1], output_content)
                output.__init__(output_size[0], output_size[1], output_content)
                self._arithmetic_operations_preserve_attributes(output, other_order=0)
                return output
            else:
                return NotImplemented
                
        def __imul__(self, other):
            if isinstance(other, Number):
                self._arithmetic_operations_assert_attributes(other, other_order=0)
                self.content.__imul__(other)
                return self
            elif isinstance(other, backend.Vector.Type()):
                self._arithmetic_operations_assert_attributes(other, other_order=1)
                self.content.__imul__(other.content)
                return self
            elif isinstance(other, backend.Function.Type()):
                return self.__imul__(other.vector())
            else:
                return NotImplemented
                
        def __truediv__(self, other):
            if isinstance(other, Number):
                self._arithmetic_operations_assert_attributes(other, other_order=0)
                output_content = self.content.__truediv__(other)
                output_size = (self.M, self.N)
                output = Matrix_Class.__new__(type(self), output_size[0], output_size[1], output_content)
                output.__init__(output_size[0], output_size[1], output_content)
                self._arithmetic_operations_preserve_attributes(output, other_order=0)
                return output
            else:
                return NotImplemented
                
        def __itruediv__(self, other):
            if isinstance(other, Number):
                self._arithmetic_operations_assert_attributes(other, other_order=0)
                self.content.__itruediv__(other)
                return self
            else:
                return NotImplemented
            
        def _arithmetic_operations_assert_attributes(self, other, other_order=2):
            assert other_order in (0, 1, 2)
            if other_order == 2:
                assert self.M == other.M
                assert self.N == other.N
                assert self._component_name_to_basis_component_index == other._component_name_to_basis_component_index
                assert self._component_name_to_basis_component_length == other._component_name_to_basis_component_length
            elif other_order == 1:
                assert self.N == other.N
                assert self._component_name_to_basis_component_index[1] == other._component_name_to_basis_component_index
                assert self._component_name_to_basis_component_length[1] == other._component_name_to_basis_component_length
                
        def _arithmetic_operations_preserve_attributes(self, output, other_order=2):
            assert other_order in (0, 1, 2)
            if other_order == 0 or other_order == 2:
                output._component_name_to_basis_component_index = self._component_name_to_basis_component_index
                output._component_name_to_basis_component_length = self._component_name_to_basis_component_length
            elif other_order == 1:
                output._component_name_to_basis_component_index = self._component_name_to_basis_component_index[0]
                output._component_name_to_basis_component_length = self._component_name_to_basis_component_length[0]
        
        def __str__(self):
            return str(self.content)
            
    return Matrix_Class
