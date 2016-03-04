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

# Hide the implementation of the storage of online data structures, with respect to the affine expansion index q.
# Both vectors over q (e.g. to store reduced order vectors/matrices) and matrices over q (e.g. to store the products
# of operators representors, as required for the efficient evaluation of error estimators)
# Requires: access with operator[]
from numpy import empty as AffineExpansionOnlineStorageContent_Base
from numpy import nditer as AffineExpansionOnlineStorageContent_Iterator
class AffineExpansionOnlineStorage(object):
    def __init__(self, *args):
        self._content = None
        if args:
            self._content = AffineExpansionOnlineStorageContent_Base(args, dtype=object)
    
    def load(self, directory, filename):
        if self._content: # avoid loading multiple times
            return False
        if io_utils.exists_numpy_file(directory, filename):
            self._content = io_utils.load_numpy_file(directory, filename)
            return True
        else:
            return False
        
    def save(self, directory, filename):
        io_utils.save_numpy_file(self._content, directory, filename)
        
    def __getitem__(self, key):
        if \
            isinstance(key, slice) \
                or \
            isinstance(key, tuple) and isinstance(key[0], slice) \
        : # return the subtensors of size "key" for every element in content. (e.g. submatrices [1:5,1:5] of the affine expansion of A)
            output = AffineExpansionOnlineStorage(*self._content.shape)
            it = AffineExpansionOnlineStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
            while not it.finished:
                output[it.multi_index] = self._content[it.multi_index][key]
                it.iternext()
            return output
        else: # return the element at position "key" in the storage (e.g. q-th matrix in the affine expansion of A, q = 1 ... Qa)
            return self._content[key]
        
    def __setitem__(self, key, item):
        assert not isinstance(key, slice) # only able to set the element at position "key" in the storage
        self._content[key] = item
        
    def __len__(self):
        if self.order() == 1: # for 1D arrays
            return self._content.size
        else:
            raise RuntimeError("Should not call len for tensors of dimension 2 or higher")
    
    def order(self):
        return len(self._content.shape)
            
#  @}
########################### end - ONLINE STAGE - end ########################### 

