# Copyright (C) 2015-2018 by the RBniCS authors
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

import os
from collections import OrderedDict
from numbers import Number
from numpy import empty as AffineExpansionStorageContent_Base
from numpy import nditer as AffineExpansionStorageContent_Iterator
from rbnics.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage, BasisFunctionsMatrix as AbstractBasisFunctionsMatrix, FunctionsList as AbstractFunctionsList
from rbnics.backends.online.basic.wrapping import slice_to_array
from rbnics.utils.decorators import overload, tuple_of
from rbnics.utils.io import Folders, OnlineSizeDict, TextIO as ContentItemShapeIO, TextIO as ContentItemTypeIO, TextIO as ContentShapeIO, TextIO as DictIO, TextIO as ScalarContentIO

def AffineExpansionStorage(backend, wrapping):
    class _AffineExpansionStorage(AbstractAffineExpansionStorage):
        def __init__(self, arg1, arg2):
            self._content = None
            self._precomputed_slices = dict() # from tuple to AffineExpansionStorage
            self._recursive = False
            self._smallest_key = None
            self._previous_key = None
            self._largest_key = None
            # Auxiliary storage for __getitem__ slicing
            self._basis_component_index_to_component_name = None # will be filled in in __setitem__, if required
            self._component_name_to_basis_component_index = None # will be filled in in __setitem__, if required
            self._component_name_to_basis_component_length = None # will be filled in in __setitem__, if required
            # Initialize arguments from inputs
            self._init(arg1, arg2)
        
        @overload(lambda cls: cls, None)
        def _init(self, arg1, arg2):
            self._recursive = True
            self._content = arg1._content
            self._precomputed_slices = arg1._precomputed_slices
            
        @overload((tuple_of(backend.Matrix.Type()), tuple_of(backend.Vector.Type())), None)
        def _init(self, arg1, arg2):
            self._recursive = False
            self._content = AffineExpansionStorageContent_Base((len(arg1), ), dtype=object)
            self._smallest_key = 0
            self._largest_key = len(arg1) - 1
            for (i, arg1i) in enumerate(arg1):
                self[i] = arg1i
            
        @overload(int, None)
        def _init(self, arg1, arg2):
            self._recursive = False
            self._content = AffineExpansionStorageContent_Base((arg1, ), dtype=object)
            self._smallest_key = 0
            self._largest_key = arg1 - 1
        
        @overload(int, int)
        def _init(self, arg1, arg2):
            self._recursive = False
            self._content = AffineExpansionStorageContent_Base((arg1, arg2), dtype=object)
            self._smallest_key = (0, 0)
            self._largest_key = (arg1 - 1, arg2 - 1)
            
        def save(self, directory, filename):
            assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
            # Get full directory name
            full_directory = Folders.Folder(os.path.join(str(directory), filename))
            full_directory.create()
            # Save content shape
            self._save_content_shape(full_directory)
            # Exit in the trivial case of empty affine expansion
            if self._content.size is 0:
                return
            # Initialize iterator
            it = AffineExpansionStorageContent_Iterator(self._content, flags=["c_index", "multi_index", "refs_ok"], op_flags=["readonly"])
            # Save content item type and shape
            self._save_content_item_type_shape(self._content[it.multi_index], it, full_directory)
            # Save content
            self._save_content(self._content[it.multi_index], it, full_directory)
            # Save dicts
            self._save_dicts(full_directory)
            
        def _save_content_shape(self, full_directory):
            ContentShapeIO.save_file(self._content.shape, full_directory, "content_shape")
        
        @overload(backend.Matrix.Type(), AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _save_content_item_type_shape(self, item, it, full_directory):
            ContentItemTypeIO.save_file("matrix", full_directory, "content_item_type")
            ContentItemShapeIO.save_file((item.M, item.N), full_directory, "content_item_shape")
        
        @overload(backend.Vector.Type(), AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _save_content_item_type_shape(self, item, it, full_directory):
            ContentItemTypeIO.save_file("vector", full_directory, "content_item_type")
            ContentItemShapeIO.save_file(item.N, full_directory, "content_item_shape")
        
        @overload(backend.Function.Type(), AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _save_content_item_type_shape(self, item, it, full_directory):
            ContentItemTypeIO.save_file("function", full_directory, "content_item_type")
            ContentItemShapeIO.save_file(item.N, full_directory, "content_item_shape")
        
        @overload(Number, AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _save_content_item_type_shape(self, item, it, full_directory):
            ContentItemTypeIO.save_file("scalar", full_directory, "content_item_type")
            ContentItemShapeIO.save_file(None, full_directory, "content_item_shape")
        
        @overload(None, AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _save_content_item_type_shape(self, item, it, full_directory):
            ContentItemTypeIO.save_file("empty", full_directory, "content_item_type")
            ContentItemShapeIO.save_file(None, full_directory, "content_item_shape")
        
        @overload(backend.Matrix.Type(), AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _save_content(self, item, it, full_directory):
            while not it.finished:
                wrapping.tensor_save(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
                it.iternext()
        
        @overload(backend.Vector.Type(), AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _save_content(self, item, it, full_directory):
            while not it.finished:
                wrapping.tensor_save(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
                it.iternext()
        
        @overload(backend.Function.Type(), AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _save_content(self, item, it, full_directory):
            while not it.finished:
                wrapping.function_save(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
                it.iternext()
        
        @overload(Number, AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _save_content(self, item, it, full_directory):
            while not it.finished:
                ScalarContentIO.save_file(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
                it.iternext()
        
        @overload(None, AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _save_content(self, item, it, full_directory):
            pass
            
        def _save_dicts(self, full_directory):
            DictIO.save_file(self._basis_component_index_to_component_name, full_directory, "basis_component_index_to_component_name")
            DictIO.save_file(self._component_name_to_basis_component_index, full_directory, "component_name_to_basis_component_index")
            DictIO.save_file(self._component_name_to_basis_component_length, full_directory, "component_name_to_basis_component_length")
            
        def load(self, directory, filename):
            assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
            if self._content is not None: # avoid loading multiple times
                if self._content.size > 0:
                    it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
                    while not it.finished:
                        if self._content[it.multi_index] is not None: # ... but only if there is at least one element different from None
                            return False
                        it.iternext()
            # Get full directory name
            full_directory = Folders.Folder(os.path.join(str(directory), filename))
            # Load content shape
            content_shape = self._load_content_shape(full_directory)
            # Prepare content
            self._content = AffineExpansionStorageContent_Base(content_shape, dtype=object)
            # Exit in the trivial case of empty affine expansion
            if self._content.size is 0:
                return True
            # Load content item type and shape
            reference_item = self._load_content_item_type_shape(full_directory)
            # Initialize iterator
            it = AffineExpansionStorageContent_Iterator(self._content, flags=["c_index", "multi_index", "refs_ok"])
            # Load content
            self._load_content(reference_item, it, full_directory)
            # Load dicts
            self._load_dicts(full_directory)
            # Reset precomputed slices
            self._precomputed_slices = dict()
            self._prepare_trivial_precomputed_slice(reference_item)
            # Return
            return True
            
        def _load_content_shape(self, full_directory):
            assert ContentShapeIO.exists_file(full_directory, "content_shape")
            return ContentShapeIO.load_file(full_directory, "content_shape")
            
        def _load_content_item_type_shape(self, full_directory):
            assert ContentItemTypeIO.exists_file(full_directory, "content_item_type")
            content_item_type = ContentItemTypeIO.load_file(full_directory, "content_item_type")
            assert ContentItemShapeIO.exists_file(full_directory, "content_item_shape")
            assert content_item_type in ("matrix", "vector", "function", "scalar", "empty")
            if content_item_type == "matrix":
                (M, N) = ContentItemShapeIO.load_file(full_directory, "content_item_shape", globals={"OnlineSizeDict": OnlineSizeDict})
                return backend.Matrix(M, N)
            elif content_item_type == "vector":
                N = ContentItemShapeIO.load_file(full_directory, "content_item_shape", globals={"OnlineSizeDict": OnlineSizeDict})
                return backend.Vector(N)
            elif content_item_type == "function":
                N = ContentItemShapeIO.load_file(full_directory, "content_item_shape", globals={"OnlineSizeDict": OnlineSizeDict})
                return backend.Function(N)
            elif content_item_type == "scalar":
                return 0.
            elif content_item_type == "empty":
                return None
            else: # impossible to arrive here anyway thanks to the assert
                raise ValueError("Invalid content item type.")
        
        @overload(backend.Matrix.Type(), AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _load_content(self, item, it, full_directory):
            while not it.finished:
                self._content[it.multi_index] = wrapping.tensor_copy(item)
                tensor_loaded = wrapping.tensor_load(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
                assert tensor_loaded
                it.iternext()
        
        @overload(backend.Vector.Type(), AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _load_content(self, item, it, full_directory):
            while not it.finished:
                self._content[it.multi_index] = wrapping.tensor_copy(item)
                tensor_loaded = wrapping.tensor_load(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
                assert tensor_loaded
                it.iternext()
        
        @overload(backend.Function.Type(), AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _load_content(self, item, it, full_directory):
            while not it.finished:
                self._content[it.multi_index] = wrapping.function_copy(item)
                function_loaded = wrapping.function_load(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
                assert function_loaded
                it.iternext()
        
        @overload(Number, AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _load_content(self, item, it, full_directory):
            while not it.finished:
                self._content[it.multi_index] = ScalarContentIO.load_file(full_directory, "content_item_" + str(it.index))
                it.iternext()
        
        @overload(None, AffineExpansionStorageContent_Iterator, Folders.Folder)
        def _load_content(self, item, it, full_directory):
            pass
            
        def _load_dicts(self, full_directory):
            assert DictIO.exists_file(full_directory, "basis_component_index_to_component_name")
            self._basis_component_index_to_component_name = DictIO.load_file(full_directory, "basis_component_index_to_component_name", globals={"OrderedDict": OrderedDict})
            assert DictIO.exists_file(full_directory, "component_name_to_basis_component_index")
            self._component_name_to_basis_component_index = DictIO.load_file(full_directory, "component_name_to_basis_component_index", globals={"OrderedDict": OrderedDict})
            assert DictIO.exists_file(full_directory, "component_name_to_basis_component_length")
            self._component_name_to_basis_component_length = DictIO.load_file(full_directory, "component_name_to_basis_component_length", globals={"OnlineSizeDict": OnlineSizeDict})
            it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
            while not it.finished:
                if self._basis_component_index_to_component_name is not None:
                    self._content[it.multi_index]._basis_component_index_to_component_name = self._basis_component_index_to_component_name
                if self._component_name_to_basis_component_index is not None:
                    self._content[it.multi_index]._component_name_to_basis_component_index = self._component_name_to_basis_component_index
                if self._component_name_to_basis_component_length is not None:
                    self._content[it.multi_index]._component_name_to_basis_component_length = self._component_name_to_basis_component_length
                it.iternext()
        
        @overload(backend.Matrix.Type(), )
        def _prepare_trivial_precomputed_slice(self, item):
            slice_0 = tuple(range(item.shape[0]))
            slice_1 = tuple(range(item.shape[1]))
            self._precomputed_slices[(slice_0, slice_1)] = self
        
        @overload(backend.Vector.Type(), )
        def _prepare_trivial_precomputed_slice(self, item):
            slice_0 = tuple(range(item.shape[0]))
            self._precomputed_slices[(slice_0, )] = self
            
        @overload(backend.Function.Type(), )
        def _prepare_trivial_precomputed_slice(self, item):
            slice_0 = tuple(range(item.vector().shape[0]))
            self._precomputed_slices[(slice_0, )] = self
            
        @overload(Number, )
        def _prepare_trivial_precomputed_slice(self, item):
            pass
            
        @overload(AbstractFunctionsList, )
        def _prepare_trivial_precomputed_slice(self, item):
            pass
            
        @overload(AbstractBasisFunctionsMatrix, )
        def _prepare_trivial_precomputed_slice(self, item):
            pass
            
        @overload(None, )
        def _prepare_trivial_precomputed_slice(self, item):
            pass
            
        @overload((slice, tuple_of(slice)), )
        def __getitem__(self, key):
            """
            return the subtensors of size "key" for every element in content. (e.g. submatrices [1:5,1:5] of the affine expansion of A)
            """
            slices = slice_to_array(self, key, self._component_name_to_basis_component_length, self._component_name_to_basis_component_index)
            
            if slices in self._precomputed_slices:
                return self._precomputed_slices[slices]
            else:
                output = _AffineExpansionStorage.__new__(type(self), *self._content.shape)
                output.__init__(*self._content.shape)
                it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
                while not it.finished:
                    item = self._content[it.multi_index]
                    # Slice content and assign
                    output[it.multi_index] = self._do_slicing(item, key)
                    # Increment
                    it.iternext()
                self._precomputed_slices[slices] = output
                return output
        
        @overload((int, tuple_of(int)), )
        def __getitem__(self, key):
            """
            return the element at position "key" in the storage (e.g. q-th matrix in the affine expansion of A, q = 1 ... Qa)
            """
            return self._content[key]
            
        @overload(backend.Matrix.Type(), (slice, tuple_of(slice)))
        def _do_slicing(self, item, key):
            return item[key]
        
        @overload(backend.Vector.Type(), (slice, tuple_of(slice)))
        def _do_slicing(self, item, key):
            return item[key]
            
        @overload(backend.Function.Type(), (slice, tuple_of(slice)))
        def _do_slicing(self, item, key):
            return backend.Function(item.vector()[key])
            
        def __setitem__(self, key, item):
            assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
            assert not isinstance(key, slice) # only able to set the element at position "key" in the storage
            # Check that __getitem__ is not random acces but called for increasing key and store current key
            self._assert_setitem_order(key)
            self._update_previous_key(key)
            # Store item
            self._content[key] = item
            # Reset attributes related to basis functions matrix if the size has changed
            if key == self._smallest_key: # this assumes that __getitem__ is not random acces but called for increasing key
                self._basis_component_index_to_component_name = None
                self._component_name_to_basis_component_index = None
                self._component_name_to_basis_component_length = None
            # Also store attributes related to basis functions matrix for __getitem__ slicing
            assert isinstance(item, (
                backend.Matrix.Type(),          # output e.g. of Z^T*A*Z
                backend.Vector.Type(),          # output e.g. of Z^T*F
                backend.Function.Type(),        # for initial conditions of unsteady problems
                Number,                         # output of Riesz_F^T*X*Riesz_F
                AbstractFunctionsList,          # auxiliary storage of Riesz representors
                AbstractBasisFunctionsMatrix    # auxiliary storage of Riesz representors
            ))
            if isinstance(item, backend.Function.Type()):
                item = item.vector()
            if isinstance(item, (backend.Matrix.Type(), backend.Vector.Type(), AbstractBasisFunctionsMatrix)):
                assert (self._basis_component_index_to_component_name is None) == (self._component_name_to_basis_component_index is None)
                assert (self._component_name_to_basis_component_index is None) == (self._component_name_to_basis_component_length is None)
                if self._basis_component_index_to_component_name is None:
                    self._basis_component_index_to_component_name = item._basis_component_index_to_component_name
                    self._component_name_to_basis_component_index = item._component_name_to_basis_component_index
                    self._component_name_to_basis_component_length = item._component_name_to_basis_component_length
                else:
                    assert self._basis_component_index_to_component_name == item._basis_component_index_to_component_name
                    assert self._component_name_to_basis_component_index == item._component_name_to_basis_component_index
                    assert self._component_name_to_basis_component_length == item._component_name_to_basis_component_length
            else:
                assert self._basis_component_index_to_component_name is None
                assert self._component_name_to_basis_component_index is None
                assert self._component_name_to_basis_component_length is None
            # Reset and prepare precomputed slices
            if key == self._largest_key: # this assumes that __getitem__ is not random acces but called for increasing key
                self._precomputed_slices = dict()
                self._prepare_trivial_precomputed_slice(item)
             
        @overload(int)
        def _assert_setitem_order(self, current_key):
            if self._previous_key is None:
                assert current_key == 0
            else:
                assert current_key == (self._previous_key + 1) % (self._largest_key + 1)
                
        @overload(int, int)
        def _assert_setitem_order(self, current_key_0, current_key_1):
            if self._previous_key is None:
                assert current_key_0 == 0
                assert current_key_1 == 0
            else:
                expected_key_1 = (self._previous_key[1] + 1) % (self._largest_key[1] + 1)
                if expected_key_1 is 0:
                    expected_key_0 = (self._previous_key[0] + 1) % (self._largest_key[0] + 1)
                else:
                    expected_key_0 = self._previous_key[0]
                assert current_key_0 == expected_key_0
                assert current_key_1 == expected_key_1
                
        @overload(tuple_of(int))
        def _assert_setitem_order(self, current_key):
            self._assert_setitem_order(*current_key)
            
        @overload(int)
        def _update_previous_key(self, current_key):
            self._previous_key = current_key
            
        @overload(int, int)
        def _update_previous_key(self, current_key_0, current_key_1):
            self._previous_key = (current_key_0, current_key_1)
                
        @overload(tuple_of(int))
        def _update_previous_key(self, current_key):
            self._update_previous_key(*current_key)
            
        def __iter__(self):
            return AffineExpansionStorageContent_Iterator(self._content, flags=["refs_ok"], op_flags=["readonly"])
            
        def __len__(self):
            assert self.order() == 1
            return self._content.size
        
        def order(self):
            assert self._content is not None
            return len(self._content.shape)
            
    return _AffineExpansionStorage
