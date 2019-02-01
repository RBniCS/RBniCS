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

def Vector(backend, wrapping, VectorBaseType):
    class Vector_Class(object):
        def __init__(self, N, content=None):
            assert isinstance(N, (int, dict))
            if isinstance(N, dict):
                N_sum = sum(N.values())
            else:
                N_sum = N
            self.N = N
            if content is None:
                self.content = VectorBaseType(N_sum)
            else:
                self.content = content
            # Auxiliary attributes related to basis functions matrix
            if isinstance(N, dict):
                if len(N) > 1:
                    # ordering (stored by OnlineSizeDict, which inherits from OrderedDict) is important in the definition of attributes
                    assert isinstance(N, OnlineSizeDict)
                else:
                    self.N = N = OnlineSizeDict(N)
                self._component_name_to_basis_component_index = ComponentNameToBasisComponentIndexDict()
                self._component_name_to_basis_component_length = OnlineSizeDict()
                for (component_index, (component_name, component_length)) in enumerate(N.items()):
                    self._component_name_to_basis_component_index[component_name] = component_index
                    self._component_name_to_basis_component_length[component_name] = component_length
            else:
                self._component_name_to_basis_component_index = None
                self._component_name_to_basis_component_length = None
            
        def __getitem__(self, key):
            if (
                isinstance(key, slice)  # vector[:5]
                    or
                isinstance(key, (list, tuple)) # vector[[0, 1, 2, 3, 4]]
            ):
                if isinstance(key, slice): # vector[:5]
                    output_content = self.content[wrapping.Slicer(slice_to_array(self, key, self._component_name_to_basis_component_length, self._component_name_to_basis_component_index))]
                    output_size = slice_to_size(self, key, self._component_name_to_basis_component_length)
                elif isinstance(key, (list, tuple)): # vector[[0, 1, 2, 3, 4]]
                    output_content = self.content[wrapping.Slicer(key)]
                    output_size = (len(key), )
                # Prepare output
                assert len(output_size) == 1
                output = Vector_Class.__new__(type(self), output_size[0], output_content)
                output.__init__(output_size[0], output_content)
                # Preserve auxiliary attributes related to basis functions matrix
                output._component_name_to_basis_component_index = self._component_name_to_basis_component_index
                if self._component_name_to_basis_component_length is None:
                    output._component_name_to_basis_component_length = None
                else:
                    if isinstance(key, slice): # vector[:5]
                        output._component_name_to_basis_component_length = output_size[0]
                    elif isinstance(key, (list, tuple)): # vector[[0, 1, 2, 3, 4]]
                        if len(self._component_name_to_basis_component_length) == 1:
                            for (component_name, _) in self._component_name_to_basis_component_length.items():
                                break
                            component_name_to_basis_component_length = OnlineSizeDict()
                            component_name_to_basis_component_length[component_name] = len(key)
                            output._component_name_to_basis_component_length = component_name_to_basis_component_length
                        else:
                            raise NotImplementedError("Vector.__getitem__ with list or tuple input arguments has not been implemented yet for the case of multiple components")
                return output
            elif isinstance(key, int): # vector[5]
                output = self.content[key]
                assert isinstance(output, Number)
                return output
            else:
                raise TypeError("Unsupported key type in Vector.__getitem__")
                
        def __setitem__(self, key, value):
            if (
                isinstance(key, slice)  # vector[:5]
                    or
                isinstance(key, (list, tuple)) # vector[[0, 1, 2, 3, 4]]
            ):
                if isinstance(key, slice): # vector[:5]
                    converted_key = wrapping.Slicer(slice_to_array(self, key, self._component_name_to_basis_component_length, self._component_name_to_basis_component_index))
                elif isinstance(key, (list, tuple)): # vector[[0, 1, 2, 3, 4]]
                    converted_key = wrapping.Slicer(key)
                if isinstance(value, type(self)):
                    value = value.content
                self.content[converted_key] = value
            elif isinstance(key, int): # vector[5]
                self.content[key] = value
            else:
                raise TypeError("Unsupported key type in Vector.__setitem__")
                
        def __abs__(self):
            self._arithmetic_operations_assert_attributes(None, other_order=0)
            output_content = self.content.__abs__()
            output_size = self.N
            output = Vector_Class.__new__(type(self), output_size, output_content)
            output.__init__(output_size, output_content)
            self._arithmetic_operations_preserve_attributes(output, other_order=0)
            return output
            
        def __neg__(self):
            self._arithmetic_operations_assert_attributes(None, other_order=0)
            output_content = self.content.__neg__()
            output_size = self.N
            output = Vector_Class.__new__(type(self), output_size, output_content)
            output.__init__(output_size, output_content)
            self._arithmetic_operations_preserve_attributes(output, other_order=0)
            return output
            
        def __add__(self, other):
            if isinstance(other, type(self)):
                self._arithmetic_operations_assert_attributes(other)
                output_content = self.content.__add__(other.content)
                output_size = self.N
                output = Vector_Class.__new__(type(self), output_size, output_content)
                output.__init__(output_size, output_content)
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
                output_size = self.N
                output = Vector_Class.__new__(type(self), output_size, output_content)
                output.__init__(output_size, output_content)
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
                output_size = self.N
                output = Vector_Class.__new__(type(self), output_size, output_content)
                output.__init__(output_size, output_content)
                self._arithmetic_operations_preserve_attributes(output, other_order=0)
                return output
            else:
                return NotImplemented
            
        def __rmul__(self, other):
            if isinstance(other, Number):
                self._arithmetic_operations_assert_attributes(other, other_order=0)
                output_content = self.content.__rmul__(other)
                output_size = self.N
                output = Vector_Class.__new__(type(self), output_size, output_content)
                output.__init__(output_size, output_content)
                self._arithmetic_operations_preserve_attributes(output, other_order=0)
                return output
            else:
                return NotImplemented
                
        def __imul__(self, other):
            if isinstance(other, Number):
                self._arithmetic_operations_assert_attributes(other, other_order=0)
                self.content.__imul__(other)
                return self
            else:
                return NotImplemented
                
        def __truediv__(self, other):
            if isinstance(other, Number):
                self._arithmetic_operations_assert_attributes(other, other_order=0)
                output_content = self.content.__truediv__(other)
                output_size = self.N
                output = Vector_Class.__new__(type(self), output_size, output_content)
                output.__init__(output_size, output_content)
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
        
        def _arithmetic_operations_assert_attributes(self, other, other_order=1):
            assert other_order in (0, 1)
            if other_order == 1:
                assert self.N == other.N
                assert self._component_name_to_basis_component_index == other._component_name_to_basis_component_index
                assert self._component_name_to_basis_component_length == other._component_name_to_basis_component_length
        
        def _arithmetic_operations_preserve_attributes(self, output, other_order=1):
            assert other_order in (0, 1)
            output._component_name_to_basis_component_index = self._component_name_to_basis_component_index
            output._component_name_to_basis_component_length = self._component_name_to_basis_component_length
        
        def __str__(self):
            return str(self.content)
            
        def __iter__(self):
            return self.content.__iter__()
            
    return Vector_Class
