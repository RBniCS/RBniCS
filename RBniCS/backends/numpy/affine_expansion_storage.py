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
## @file affine_expansion_online_storage.py
#  @brief Type for storing online quantities related to an affine expansion
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import empty as AffineExpansionStorageContent_Base
from numpy import nditer as AffineExpansionStorageContent_Iterator
from numpy import asmatrix as AffineExpansionStorageContent_AsMatrix
from RBniCS.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage, FunctionsList as AbstractFunctionsList
from RBniCS.backends.numpy.matrix import Matrix as OnlineMatrix
from RBniCS.backends.numpy.vector import Vector as OnlineVector
from RBniCS.backends.numpy.wrapping_utils import slice_to_array, slice_to_size
from RBniCS.utils.io import NumpyIO as AffineExpansionStorageContent_IO, Folders, PickleIO as ContentSizeIO, PickleIO as ContentTypeIO, PickleIO as DictIO
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override

@Extends(AbstractAffineExpansionStorage)
@BackendFor("numpy", inputs=((int, AbstractAffineExpansionStorage), (int, None)))
class AffineExpansionStorage(AbstractAffineExpansionStorage):
    @override
    def __init__(self, arg1, arg2=None):
        self._content = None
        self._content_as_matrix = None
        self._precomputed_slices = dict() # from tuple to AffineExpansionStorage
        self._recursive = False
        # Carry out initialization
        assert (
            (isinstance(arg1, int) and isinstance(arg1, AbstractAffineExpansionStorage))
                or
            (isinstance(arg2, int) or arg2 is None)
        )
        if isinstance(arg1, AbstractAffineExpansionStorage):
            assert arg2 is None
            self._recursive = True
            self._content = arg1._content
            self._content_as_matrix = arg1._content_as_matrix
            self._precomputed_slices = arg1._precomputed_slices
        elif isinstance(arg1, int):
            if arg2 is None:
                self._recursive = False
                self._content = AffineExpansionStorageContent_Base((arg1, ), dtype=object)
            else:
                assert isinstance(arg2, int)
                self._recursive = False
                self._content = AffineExpansionStorageContent_Base((arg1, arg2), dtype=object)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid argument to AffineExpansionStorage")
        # Auxiliary storage for __getitem__ slicing
        self._basis_component_index_to_component_name = None # will be filled in in __setitem__, if required
        self._component_name_to_basis_component_index = None # will be filled in in __setitem__, if required
        self._component_name_to_basis_component_length = None # will be filled in in __setitem__, if required
        
    @override
    def save(self, directory, filename):
        assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
        # Get full directory name
        full_directory = Folders.Folder(directory + "/" + filename)
        full_directory.create()
        # Save affine expansion
        AffineExpansionStorageContent_IO.save_file(self._content, full_directory, "content")
        # Save dicts
        DictIO.save_file(self._basis_component_index_to_component_name, full_directory, "basis_component_index_to_component_name")
        DictIO.save_file(self._component_name_to_basis_component_index, full_directory, "component_name_to_basis_component_index")
        DictIO.save_file(self._component_name_to_basis_component_length, full_directory, "component_name_to_basis_component_length")
        # Save size
        it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
        item = self._content[it.multi_index]
        assert isinstance(item, (OnlineMatrix.Type(), OnlineVector.Type(), float)) or item is None # these are the only types which we are interested in saving
        if isinstance(item, OnlineMatrix.Type()):
            ContentTypeIO.save_file("matrix", full_directory, "content_type")
            ContentSizeIO.save_file((item.M, item.N), full_directory, "content_size")
        elif isinstance(item, OnlineVector.Type()):
            ContentTypeIO.save_file("vector", full_directory, "content_type")
            ContentSizeIO.save_file(item.N, full_directory, "content_size")
        elif isinstance(item, float):
            ContentTypeIO.save_file("scalar", full_directory, "content_type")
            ContentSizeIO.save_file(None, full_directory, "content_size")
        elif item is None:
            ContentTypeIO.save_file("empty", full_directory, "content_type")
            ContentSizeIO.save_file(None, full_directory, "content_size")            
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid content type.")
    
    @override
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
        # Load affine expansion
        assert AffineExpansionStorageContent_IO.exists_file(full_directory, "content")
        self._content = AffineExpansionStorageContent_IO.load_file(full_directory, "content")
        # Load dicts
        assert DictIO.exists_file(full_directory, "basis_component_index_to_component_name")
        self._basis_component_index_to_component_name = DictIO.load_file(full_directory, "basis_component_index_to_component_name")
        assert DictIO.exists_file(full_directory, "component_name_to_basis_component_index")
        self._component_name_to_basis_component_index = DictIO.load_file(full_directory, "component_name_to_basis_component_index")
        assert DictIO.exists_file(full_directory, "component_name_to_basis_component_length")
        self._component_name_to_basis_component_length = DictIO.load_file(full_directory, "component_name_to_basis_component_length")
        # Load size
        assert ContentTypeIO.exists_file(full_directory, "content_type")
        content_type = ContentTypeIO.load_file(full_directory, "content_type")
        assert ContentSizeIO.exists_file(full_directory, "content_size")
        assert content_type in ("matrix", "vector", "scalar", "empty")
        if content_type == "matrix":
            (M, N) = ContentSizeIO.load_file(full_directory, "content_size")
            it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
            while not it.finished:
                self._content[it.multi_index].M = M
                self._content[it.multi_index].N = N
                it.iternext()
        elif content_type == "vector":
            N = ContentSizeIO.load_file(full_directory, "content_size")
            it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
            while not it.finished:
                self._content[it.multi_index].N = N
                it.iternext()
        elif content_type == "scalar":
            pass # nothing to be done
        elif content_type == "empty":
            pass # nothing to be done
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid content type.")
        # Create internal copy as matrix
        self._content_as_matrix = None
        self.as_matrix()
        # Reset precomputed slices
        it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
        self._prepare_trivial_precomputed_slice(self._content[it.multi_index])
        # Return
        return True
            
    def _prepare_trivial_precomputed_slice(self, item):
        # Reset precomputed slices
        self._precomputed_slices = dict()
        # Prepare trivial precomputed slice
        if isinstance(item, OnlineMatrix.Type()):
            slice_0 = tuple(range(item.shape[0]))
            slice_1 = tuple(range(item.shape[1]))
            self._precomputed_slices[(slice_0, slice_1)] = self
        elif isinstance(item, OnlineVector.Type()):
            slice_0 = tuple(range(item.shape[0]))
            self._precomputed_slices[(slice_0, )] = self
        
    def as_matrix(self):
        if self._content_as_matrix is None:
            self._content_as_matrix = AffineExpansionStorageContent_AsMatrix(self._content)
        return self._content_as_matrix
    
    @override
    def __getitem__(self, key):
        if (
            isinstance(key, slice)
                or
            ( isinstance(key, tuple) and isinstance(key[0], slice) )
        ): # return the subtensors of size "key" for every element in content. (e.g. submatrices [1:5,1:5] of the affine expansion of A)
            
            slices = slice_to_array(key, self._component_name_to_basis_component_length, self._component_name_to_basis_component_index)
            
            if slices in self._precomputed_slices:
                return self._precomputed_slices[slices]
            else:
                output = AffineExpansionStorage(*self._content.shape)
                output_content_size = slice_to_size(key, self._component_name_to_basis_component_length)
                it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
                while not it.finished:
                    item = self._content[it.multi_index]
                    
                    # Slice content
                    assert isinstance(item, (OnlineMatrix.Type(), OnlineVector.Type()))
                    sliced_item = item[slices]
                        
                    # Copy dicts
                    sliced_item._basis_component_index_to_component_name = self._basis_component_index_to_component_name
                    sliced_item._component_name_to_basis_component_index = self._component_name_to_basis_component_index
                    sliced_item._component_name_to_basis_component_length = self._component_name_to_basis_component_length
                    
                    # Copy size
                    if isinstance(item, OnlineMatrix.Type()):
                        assert len(output_content_size) == 2
                        sliced_item.M = output_content_size[0]
                        sliced_item.N = output_content_size[1]
                    elif isinstance(item, OnlineVector.Type()):
                        assert len(output_content_size) == 1
                        sliced_item.N = output_content_size[0]
                    else: # impossible to arrive here anyway thanks to the assert
                        raise AssertionError("Invalid item in slicing.")
                    
                    # Assign
                    output[it.multi_index] = sliced_item
                    
                    # Increment
                    it.iternext()
                self._precomputed_slices[slices] = output
                return output
                
        else: # return the element at position "key" in the storage (e.g. q-th matrix in the affine expansion of A, q = 1 ... Qa)
            return self._content[key]
        
    @override
    def __setitem__(self, key, item):
        assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
        assert not isinstance(key, slice) # only able to set the element at position "key" in the storage
        self._content[key] = item
        # Also store component_name_to_basis_component_* for __getitem__ slicing
        assert isinstance(item, (
            OnlineMatrix.Type(),  # output e.g. of Z^T*A*Z
            OnlineVector.Type(),  # output e.g. of Z^T*F
            float,                # output of Riesz_F^T*X*Riesz_F
            AbstractFunctionsList # auxiliary storage of Riesz representors
        ))
        assert hasattr(item, "_basis_component_index_to_component_name") == hasattr(item, "_component_name_to_basis_component_index")
        assert hasattr(item, "_component_name_to_basis_component_index") == hasattr(item, "_component_name_to_basis_component_length")
        if hasattr(item, "_component_name_to_basis_component_index"): # temporarily added by transpose() method
            assert isinstance(item, (OnlineMatrix.Type(), OnlineVector.Type()))
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
            del item._basis_component_index_to_component_name # cleanup temporary addition
            del item._component_name_to_basis_component_index # cleanup temporary addition
            del item._component_name_to_basis_component_length # cleanup temporary addition
        else:
            assert self._basis_component_index_to_component_name is None
            assert self._component_name_to_basis_component_index is None
            assert self._component_name_to_basis_component_length is None
        # Reset internal copies
        self._content_as_matrix = None
        # Reset and prepare precomputed slices
        self._prepare_trivial_precomputed_slice(item)
        
    @override
    def __iter__(self):
        return AffineExpansionStorageContent_Iterator(self._content, flags=["refs_ok"], op_flags=["readonly"])
        
    @override
    def __len__(self):
        assert self.order() == 1
        return self._content.size
    
    def order(self):
        assert self._content is not None
        return len(self._content.shape)
        
