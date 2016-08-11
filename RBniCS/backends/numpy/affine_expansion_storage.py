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
from RBniCS.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage
from RBniCS.utils.io import NumpyIO as AffineExpansionStorageContent_IO
from RBniCS.utils.decorators import BackendFor, Extends, override

@Extends(AbstractAffineExpansionStorage)
@BackendFor("NumPy", inputs=((list, int, AbstractAffineExpansionStorage), ))
class AffineExpansionStorage(AbstractAffineExpansionStorage):
    @override
    def __init__(self, args):
        self._content = None
        self._content_as_matrix = None
        self._precomputed_slices = dict() # from tuple to AffineExpansionStorage
        self._recursive = False
        self.init(args)
        
    @override
    def init(self, args):
        assert isinstance(args, (list, int, AbstractAffineExpansionStorage))
        if isinstance(args, AbstractAffineExpansionStorage):
            self._recursive = True
            self._content = args[0]._content
            self._content_as_matrix = args[0]._content_as_matrix
            self._precomputed_slices = args[0]._precomputed_slices
        elif isinstance(args, int):
            self._recursive = False
            self._content = AffineExpansionStorageContent_Base((args, ), dtype=object)
        elif isinstance(args, list):
            for i in args:
                assert isinstance(i, int)
            self._recursive = False
            self._content = AffineExpansionStorageContent_Base(args, dtype=object)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid argument to AffineExpansionStorage")
    
    @override
    def load(self, directory, filename):
        assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
        if self._content is not None: # avoid loading multiple times
            it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
            while not it.finished:
                if self._content[it.multi_index] is not None: # ... but only if there is at least one element different from None
                    return False
                it.iternext()
        if AffineExpansionStorageContent_IO.exists_file(directory, filename):
            self._content = AffineExpansionStorageContent_IO.load_file(directory, filename)
            # Create internal copy as matrix
            self._content_as_matrix = None
            self.as_matrix()
            # Reset precomputed slices
            self._precomputed_slices = dict()
            return True
        else:
            return False
        
    @override
    def save(self, directory, filename):
        assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
        AffineExpansionStorageContent_IO.save_file(self._content, directory, filename)
        
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
            
            if isinstance(key, slice):
                key = (key,)
                
            assert isinstance(key, tuple)
            assert isinstance(key[0], slice)
            
            dict_key = list()
            for slice_ in key:
                assert slice_.start is None 
                assert slice_.step is None
                dict_key.append(slice_.stop)
            dict_key = tuple(dict_key)
            
            if dict_key in self._precomputed_slices:
                return self._precomputed_slices[dict_key]
                            
            it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
            
            is_slice_equal_to_full_tensor = True
            for slice_ in key:
                assert slice_.start is None 
                assert slice_.step is None
                assert slice_.stop <= self._content[it.multi_index].shape[i]
                if slice_.stop < self._content[it.multi_index].shape[i]:
                    is_slice_equal_to_full_tensor = False
            if is_slice_equal_to_full_tensor:
                self._precomputed_slices[dict_key] = self
                return self
            
            output = AffineExpansionStorage(*self._content.shape)
            while not it.finished:
                output[it.multi_index] = self._content[it.multi_index][key]
                it.iternext()
            self._precomputed_slices[dict_key] = output
            return output
        else: # return the element at position "key" in the storage (e.g. q-th matrix in the affine expansion of A, q = 1 ... Qa)
            return self._content[key]
        
    @override
    def __setitem__(self, key, item):
        assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
        assert not isinstance(key, slice) # only able to set the element at position "key" in the storage
        self._content[key] = item
        # Reset internal copies
        self._content_as_matrix = None
        self._precomputed_slices = dict()
        
    @override
    def __iter__(self):
        return AffineExpansionStorageContent_Iterator(self._content, op_flags=["readonly"])
        
    @override
    def __len__(self):
        assert self.order() == 1
        return self._content.size
    
    def order(self):
        assert self._content is not None
        return len(self._content.shape)
        
