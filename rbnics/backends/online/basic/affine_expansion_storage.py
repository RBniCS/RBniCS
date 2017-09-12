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

from numpy import empty as AffineExpansionStorageContent_Base
from numpy import nditer as AffineExpansionStorageContent_Iterator
from rbnics.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage, BasisFunctionsMatrix as AbstractBasisFunctionsMatrix, FunctionsList as AbstractFunctionsList
from rbnics.backends.online.basic.wrapping import slice_to_array
from rbnics.utils.io import Folders, PickleIO as ContentItemShapeIO, PickleIO as ContentItemTypeIO, PickleIO as ContentShapeIO, PickleIO as DictIO, PickleIO as ScalarContentIO
from rbnics.utils.decorators import Extends

@Extends(AbstractAffineExpansionStorage)
class AffineExpansionStorage(AbstractAffineExpansionStorage):
    def __init__(self, arg1, arg2, backend, wrapping):
        self.backend = backend
        self.wrapping = wrapping
        self._content = None
        self._precomputed_slices = dict() # from tuple to AffineExpansionStorage
        self._recursive = False
        self._largest_key = None
        # Carry out initialization
        assert (
            isinstance(arg1, (int, tuple, AbstractAffineExpansionStorage))
                or
            (isinstance(arg2, int) or arg2 is None)
        )
        if isinstance(arg1, AbstractAffineExpansionStorage):
            assert arg2 is None
            self._recursive = True
            self._content = arg1._content
            self._precomputed_slices = arg1._precomputed_slices
        elif isinstance(arg1, tuple):
            assert all([isinstance(arg1i, (backend.Matrix.Type(), backend.Vector.Type())) for arg1i in arg1])
            assert arg2 is None
            self._recursive = False
            self._content = AffineExpansionStorageContent_Base((len(arg1), ), dtype=object)
            self._largest_key = len(arg1) - 1
        elif isinstance(arg1, int):
            if arg2 is None:
                self._recursive = False
                self._content = AffineExpansionStorageContent_Base((arg1, ), dtype=object)
                self._largest_key = arg1 - 1
            else:
                assert isinstance(arg2, int)
                self._recursive = False
                self._content = AffineExpansionStorageContent_Base((arg1, arg2), dtype=object)
                self._largest_key = (arg1 - 1, arg2 - 1)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid argument to AffineExpansionStorage")
        # Auxiliary storage for __getitem__ slicing
        self._basis_component_index_to_component_name = None # will be filled in in __setitem__, if required
        self._component_name_to_basis_component_index = None # will be filled in in __setitem__, if required
        self._component_name_to_basis_component_length = None # will be filled in in __setitem__, if required
        # Finish copy construction, if argument is tuple
        if isinstance(arg1, tuple):
            for (i, arg1i) in enumerate(arg1):
                self[i] = arg1i
        
    def save(self, directory, filename):
        assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
        # Get full directory name
        full_directory = Folders.Folder(directory + "/" + filename)
        full_directory.create()
        # Save content item type and shape
        it = AffineExpansionStorageContent_Iterator(self._content, flags=["c_index", "multi_index", "refs_ok"], op_flags=["readonly"])
        item = self._content[it.multi_index]
        assert isinstance(item, (self.backend.Matrix.Type(), self.backend.Vector.Type(), self.backend.Function.Type(), float)) or item is None # these are the only types which we are interested in saving
        if isinstance(item, self.backend.Matrix.Type()):
            content_item_type = "matrix"
            ContentItemTypeIO.save_file(content_item_type, full_directory, "content_item_type")
            ContentItemShapeIO.save_file((item.M, item.N), full_directory, "content_item_shape")
        elif isinstance(item, self.backend.Vector.Type()):
            content_item_type = "vector"
            ContentItemTypeIO.save_file(content_item_type, full_directory, "content_item_type")
            ContentItemShapeIO.save_file(item.N, full_directory, "content_item_shape")
        elif isinstance(item, self.backend.Function.Type()):
            content_item_type = "function"
            ContentItemTypeIO.save_file(content_item_type, full_directory, "content_item_type")
            ContentItemShapeIO.save_file(item.N, full_directory, "content_item_shape")
        elif isinstance(item, float):
            content_item_type = "scalar"
            ContentItemTypeIO.save_file(content_item_type, full_directory, "content_item_type")
            ContentItemShapeIO.save_file(None, full_directory, "content_item_shape")
        elif item is None:
            content_item_type = "empty"
            ContentItemTypeIO.save_file(content_item_type, full_directory, "content_item_type")
            ContentItemShapeIO.save_file(None, full_directory, "content_item_shape")            
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid content item type.")
        # Save content shape
        ContentShapeIO.save_file(self._content.shape, full_directory, "content_shape")
        # Save content
        while not it.finished:
            if content_item_type in ("matrix", "vector"):
                self.wrapping.tensor_save(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
            elif content_item_type == "function":
                self.wrapping.function_save(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
            elif content_item_type == "scalar":
                ScalarContentIO.save_file(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
            elif content_item_type == "empty":
                pass # nothing to be done
            else: # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid content item type.")
            it.iternext()
        # Save dicts
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
        full_directory = directory + "/" + filename
        # Load content item type and shape
        assert ContentItemTypeIO.exists_file(full_directory, "content_item_type")
        content_item_type = ContentItemTypeIO.load_file(full_directory, "content_item_type")
        assert ContentItemShapeIO.exists_file(full_directory, "content_item_shape")
        assert content_item_type in ("matrix", "vector", "function", "scalar", "empty")
        if content_item_type == "matrix":
            (M, N) = ContentItemShapeIO.load_file(full_directory, "content_item_shape")
        elif content_item_type == "vector" or content_item_type == "function":
            N = ContentItemShapeIO.load_file(full_directory, "content_item_shape")
        elif content_item_type == "scalar":
            pass # nothing to be done
        elif content_item_type == "empty":
            pass # nothing to be done
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid content item type.")
        # Load content shape
        assert ContentShapeIO.exists_file(full_directory, "content_shape")
        content_shape = ContentShapeIO.load_file(full_directory, "content_shape")
        # Load content
        self._content = AffineExpansionStorageContent_Base(content_shape, dtype=object)
        it = AffineExpansionStorageContent_Iterator(self._content, flags=["c_index", "multi_index", "refs_ok"])
        while not it.finished:
            if content_item_type == "matrix":
                self._content[it.multi_index] = self.backend.Matrix(M, N)
                tensor_loaded = self.wrapping.tensor_load(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
                assert tensor_loaded
            elif content_item_type == "vector":
                self._content[it.multi_index] = self.backend.Vector(N)
                tensor_loaded = self.wrapping.tensor_load(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
                assert tensor_loaded
            elif content_item_type == "function":
                self._content[it.multi_index] = self.backend.Function(N)
                function_loaded = self.wrapping.function_load(self._content[it.multi_index], full_directory, "content_item_" + str(it.index))
                assert function_loaded
            elif content_item_type == "scalar":
                self._content[it.multi_index] = ScalarContentIO.load_file(full_directory, "content_item_" + str(it.index))
            elif content_item_type == "empty":
                pass # nothing to be done
            else: # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid content type.")
            it.iternext()
        # Load dicts
        assert DictIO.exists_file(full_directory, "basis_component_index_to_component_name")
        self._basis_component_index_to_component_name = DictIO.load_file(full_directory, "basis_component_index_to_component_name")
        assert DictIO.exists_file(full_directory, "component_name_to_basis_component_index")
        self._component_name_to_basis_component_index = DictIO.load_file(full_directory, "component_name_to_basis_component_index")
        assert DictIO.exists_file(full_directory, "component_name_to_basis_component_length")
        self._component_name_to_basis_component_length = DictIO.load_file(full_directory, "component_name_to_basis_component_length")
        it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
        while not it.finished:
            if self._basis_component_index_to_component_name is not None:
                self._content[it.multi_index]._basis_component_index_to_component_name = self._basis_component_index_to_component_name
            if self._component_name_to_basis_component_index is not None:
                self._content[it.multi_index]._component_name_to_basis_component_index = self._component_name_to_basis_component_index
            if self._component_name_to_basis_component_length is not None:
                self._content[it.multi_index]._component_name_to_basis_component_length = self._component_name_to_basis_component_length
            it.iternext()
        # Reset precomputed slices
        it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
        self._prepare_trivial_precomputed_slice(self._content[it.multi_index])
        # Return
        return True
            
    def _prepare_trivial_precomputed_slice(self, item):
        # Reset precomputed slices
        self._precomputed_slices = dict()
        # Prepare trivial precomputed slice
        if isinstance(item, self.backend.Matrix.Type()):
            slice_0 = tuple(range(item.shape[0]))
            slice_1 = tuple(range(item.shape[1]))
            self._precomputed_slices[(slice_0, slice_1)] = self
        elif isinstance(item, self.backend.Vector.Type()):
            slice_0 = tuple(range(item.shape[0]))
            self._precomputed_slices[(slice_0, )] = self
        
    def __getitem__(self, key):
        if (
            isinstance(key, slice)
                or
            ( isinstance(key, tuple) and isinstance(key[0], slice) )
        ): # return the subtensors of size "key" for every element in content. (e.g. submatrices [1:5,1:5] of the affine expansion of A)
            
            slices = slice_to_array(self, key, self._component_name_to_basis_component_length, self._component_name_to_basis_component_index)
            
            if slices in self._precomputed_slices:
                return self._precomputed_slices[slices]
            else:
                output = self.backend.AffineExpansionStorage(*self._content.shape)
                it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
                while not it.finished:
                    item = self._content[it.multi_index]
                    
                    # Slice content and assign
                    assert isinstance(item, (self.backend.Matrix.Type(), self.backend.Vector.Type(), self.backend.Function.Type()))
                    if isinstance(item, (self.backend.Matrix.Type(), self.backend.Vector.Type())):
                        output[it.multi_index] = item[key]
                    elif isinstance(item, self.backend.Function.Type()):
                        output[it.multi_index] = self.backend.Function(item.vector()[key])
                    else: # impossible to arrive here anyway thanks to the assert
                        raise AssertionError("Invalid item in slicing.")
                    
                    # Increment
                    it.iternext()
                self._precomputed_slices[slices] = output
                return output
                
        else: # return the element at position "key" in the storage (e.g. q-th matrix in the affine expansion of A, q = 1 ... Qa)
            return self._content[key]
        
    def __setitem__(self, key, item):
        assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
        assert not isinstance(key, slice) # only able to set the element at position "key" in the storage
        self._content[key] = item
        # Also store component_name_to_basis_component_* for __getitem__ slicing
        assert isinstance(item, (
            self.backend.Matrix.Type(),     # output e.g. of Z^T*A*Z
            self.backend.Vector.Type(),     # output e.g. of Z^T*F
            self.backend.Function.Type(),   # for initial conditions of unsteady problems
            float,                          # output of Riesz_F^T*X*Riesz_F
            AbstractFunctionsList,          # auxiliary storage of Riesz representors
            AbstractBasisFunctionsMatrix    # auxiliary storage of Riesz representors
        ))
        if isinstance(item, self.backend.Function.Type()):
            item = item.vector()
        assert hasattr(item, "_basis_component_index_to_component_name") == hasattr(item, "_component_name_to_basis_component_index")
        assert hasattr(item, "_component_name_to_basis_component_index") == hasattr(item, "_component_name_to_basis_component_length")
        if hasattr(item, "_component_name_to_basis_component_index"): 
            assert isinstance(item, (self.backend.Matrix.Type(), self.backend.Vector.Type(), AbstractBasisFunctionsMatrix))
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
            self._prepare_trivial_precomputed_slice(item)
        
    def __iter__(self):
        return AffineExpansionStorageContent_Iterator(self._content, flags=["refs_ok"], op_flags=["readonly"])
        
    def __len__(self):
        assert self.order() == 1
        return self._content.size
    
    def order(self):
        assert self._content is not None
        return len(self._content.shape)
        
