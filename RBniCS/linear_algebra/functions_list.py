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
## @file functions_list.py
#  @brief Type for storing a list of FE functions without storing them in a matrix
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

class FunctionsList(object):
    def __init__(self):
        self._list = []
    
    def enrich(functions):
        import collections
        if isinstance(functions, collections.Iterable): # more than one function
            self._list.extend(functions) # assume that they where already copied
        else: # one function
            self.list.append(functions.vector().copy()) # copy it explicitly
            
    def load(self, directory, filename):
        if self._list: # avoid loading multiple times
            return False
        if io_utils.exists_pickle_file(directory, filename):
            self._list = io_utils.load_pickle_file(directory, filename)
            return True
        else:
            return False
        
    def save(self, directory, filename):
        io_utils.save_pickle_file(self._list, directory, filename)
        for f in range(len(self._list)):
            full_filename = directory + "/" + filename + "_" + str(f) + ".pvd"
            if not os.path.exists(full_filename):
                file = File(filename, "compressed")
                file << self._list[f]
            
    def __getitem__(self, key):
        return self._list[key]
        
    def __iter__(self):
        return iter(self._list)
        
    def __len__(self):
        return len(self._list)

#  @}
########################### end - OFFLINE STAGE - end ########################### 

