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
## @file affine_expansion_storage.py
#  @brief Type for storing quantities related to an affine expansion
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     OFFLINE AND ONLINE COMMON INTERFACES     ########################### 
## @defgroup OfflineOnlineInterfaces Common interfaces for offline and online
#  @{

# Hide the implementation of an array with two or more indices, used to store tensors
# for error estimation. There are typically two kind of indices, either over the
# affine expansion or over the basis functions. This container will be indicized
# over the affine expansion. Its content will be another container, indicized over
# the basis functions
# Requires: access with operator[]
from numpy import empty as AffineExpansionStorageContent_Base
class AffineExpansionStorage(object):
    def __init__(self, *args):
        self._content = None
        if args:
            self._content = AffineExpansionStorageContent_Base(args, dtype=object)
            
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
        return self._content[key]
        
    def __setitem__(self, key, item):
        self._content[key] = item
        
#  @}
########################### end - OFFLINE AND ONLINE COMMON INTERFACES - end ########################### 

