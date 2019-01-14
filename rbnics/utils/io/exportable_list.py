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

from rbnics.utils.io.numpy_io import NumpyIO
from rbnics.utils.io.pickle_io import PickleIO
from rbnics.utils.io.text_io import TextIO
from rbnics.utils.decorators import list_of, overload

class ExportableList(object):
    def __init__(self, import_export_backend, original_list=None):
        self._list = list()
        if import_export_backend == "numpy":
            self._FileIO = NumpyIO
        elif import_export_backend == "pickle":
            self._FileIO = PickleIO
        elif import_export_backend == "text":
            self._FileIO = TextIO
        else:
            raise ValueError("Invalid import/export backend")
        if original_list is not None:
            self._list.extend(original_list)
    
    def append(self, element):
        self._list.append(element)
    
    @overload(list_of(object))
    def extend(self, other_list):
        self._list.extend(other_list)
        
    @overload(lambda cls: cls)
    def extend(self, other_list):
        self._list.extend(other_list._list)
            
    def save(self, directory, filename):
        self._FileIO.save_file(self._list, directory, filename)
    
    # Returns False if the list had been already imported so no further
    # action was needed.
    # Returns True if it was possible to import the list.
    # Raises an error if it was not possible to import the list.
    def load(self, directory, filename):
        if self._list: # avoid loading multiple times
            return False
        if self._FileIO.exists_file(directory, filename):
            self._list = self._FileIO.load_file(directory, filename)
            return True
        else:
            raise OSError
                         
    def __getitem__(self, key):
        return self._list[key]
        
    def __setitem__(self, key, item):
        self._list[key] = item
        
    def __iter__(self):
        return iter(self._list)
        
    def __len__(self):
        return len(self._list)
        
    def __str__(self):
        return str(self._list)
