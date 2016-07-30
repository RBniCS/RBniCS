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

###########################     ONLINE STAGE     ########################### 
## @defgroup OnlineStage Methods related to the online stage
#  @{

from numpy import empty as AffineExpansionOnlineStorageContent_Base
from numpy import nditer as AffineExpansionOnlineStorageContent_Iterator
from numpy import asmatrix as AffineExpansionOnlineStorageContent_AsMatrix
from RBniCS.utils.io import NumpyIO as AffineExpansionOnlineStorageContent_IO

# Hide the implementation of the storage of online data structures, with respect to the affine expansion index q.
# Both vectors over q (e.g. to store reduced order vectors/matrices) and matrices over q (e.g. to store the products
# of operators representors, as required for the efficient evaluation of error estimators)
# Requires: access with operator[]
class AffineExpansionOnlineStorage(object):
    def __init__(self, *args):
        self._content = None
        if args:
            self._content = AffineExpansionOnlineStorageContent_Base(args, dtype=object)
        self._content_as_matrix = None
        self._precomputed_slices = dict() # from tuple to AffineExpansionOnlineStorage
    
    def load(self, directory, filename):
        if self._content is not None: # avoid loading multiple times
            it = AffineExpansionOnlineStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
            while not it.finished:
                if self._content[it.multi_index] is not None: # ... but only if there is at least one element different from None
                    return False
                it.iternext()
        if AffineExpansionOnlineStorageContent_IO.exists_file(directory, filename):
            self._content = AffineExpansionOnlineStorageContent_IO.load_file(directory, filename)
            # Create internal copy as matrix
            self._content_as_matrix = None
            self.as_matrix()
            # Reset precomputed slices
            self._precomputed_slices = dict()
            return True
        else:
            return False
        
    def save(self, directory, filename):
        AffineExpansionOnlineStorageContent_IO.save_file(self._content, directory, filename)
    
    def as_matrix(self):
        if self._content_as_matrix is None:
            self._content_as_matrix = AffineExpansionOnlineStorageContent_AsMatrix(self._content)
        return self._content_as_matrix
    
    def __getitem__(self, key):
        if \
            isinstance(key, slice) \
                or \
            isinstance(key, tuple) and isinstance(key[0], slice) \
        : # return the subtensors of size "key" for every element in content. (e.g. submatrices [1:5,1:5] of the affine expansion of A)
            
            if isinstance(key, slice):
                key = (key,)
                
            assert isinstance(key, tuple) and isinstance(key[0], slice)
            
            dict_key = list()
            for i in range(len(key)):
                assert key[i].start is None and key[i].step is None
                dict_key.append(key[i].stop)
            dict_key = tuple(dict_key)
            
            if dict_key in self._precomputed_slices:
                return self._precomputed_slices[dict_key]
                            
            it = AffineExpansionOnlineStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
            
            is_slice_equal_to_full_tensor = True
            for i in range(len(key)):
                assert key[i].start is None and key[i].step is None
                assert key[i].stop <= self._content[it.multi_index].shape[i]
                if key[i].stop < self._content[it.multi_index].shape[i]:
                    is_slice_equal_to_full_tensor = False
            if is_slice_equal_to_full_tensor:
                self._precomputed_slices[dict_key] = self
                return self
            
            output = AffineExpansionOnlineStorage(*self._content.shape)
            while not it.finished:
                output[it.multi_index] = self._content[it.multi_index][key]
                it.iternext()
            self._precomputed_slices[dict_key] = output
            return output
        else: # return the element at position "key" in the storage (e.g. q-th matrix in the affine expansion of A, q = 1 ... Qa)
            return self._content[key]
        
    def __setitem__(self, key, item):
        assert not isinstance(key, slice) # only able to set the element at position "key" in the storage
        self._content[key] = item
        # Reset internal copies
        self._content_as_matrix = None
        self._precomputed_slices = dict()
        
    def __len__(self):
        assert self.order() == 1
        return self._content.size
    
    def order(self):
        assert self._content is not None
        return len(self._content.shape)
            
#  @}
########################### end - ONLINE STAGE - end ########################### 

