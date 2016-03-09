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
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

class ExportableList(object):
    def __init__(self, import_export_backend):
        self._list = []
        self._import_export_backend = import_export_backend
    
    def append(self, element):
        self._list.append(element)
            
    def load(self, directory, filename):
        if self._list: # avoid loading multiple times
            return False
        if self._import_export_backend == "numpy":
            if io_utils.exists_numpy_file(directory, filename):
                self._list = io_utils.load_numpy_file(directory, filename)
                return True
            else:
                return False
        elif self._import_export_backend == "pickle":
            if io_utils.exists_pickle_file(directory, filename):
                self._list = io_utils.load_pickle_file(directory, filename)
                return True
            else:
                return False
        else:
            raise RuntimeError("Invalid import/export backend")
        
    def save(self, directory, filename):
        if self._import_export_backend == "numpy":
            io_utils.save_numpy_file(self._list, directory, filename)
        elif self._import_export_backend == "pickle":
            io_utils.save_pickle_file(self._list, directory, filename)
        else:
            raise RuntimeError("Invalid import/export backend")
                 
    def __getitem__(self, key):
        return self._list[key]
        
    def __iter__(self):
        return iter(self._list)
        
    def __len__(self):
        return len(self._list)
     
#  @}
########################### end - OFFLINE STAGE - end ########################### 

