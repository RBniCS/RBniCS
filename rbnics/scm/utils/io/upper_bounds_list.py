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

import os
from rbnics.backends.online import online_copy, online_export, online_import_, OnlineVector
from rbnics.utils.decorators import list_of, overload
from rbnics.utils.io import Folders, TextIO as ItemVectorDimensionIO, TextIO as LenIO

class UpperBoundsList(list):
    def __init__(self):
        self._list = list()
    
    def append(self, element):
        self._list.append(element)
    
    @overload(list_of(OnlineVector.Type()))
    def extend(self, other_list):
        self._list.extend(other_list)
        
    @overload(lambda cls: cls)
    def extend(self, other_list):
        self._list.extend(other_list._list)
            
    def save(self, directory, filename):
        # Get full directory name
        full_directory = Folders.Folder(os.path.join(str(directory), filename))
        full_directory.create()
        # Save list length
        self._save_len(full_directory)
        # Save list item vector dimension
        self._save_item_vector_dimension(full_directory)
        # Save list
        self._save_list(full_directory)
        
    def _save_len(self, full_directory):
        LenIO.save_file(len(self._list), full_directory, "len")
        
    def _save_item_vector_dimension(self, full_directory):
        ItemVectorDimensionIO.save_file(self._list[0].N, full_directory, "item_vector_dimension")
        
    def _save_list(self, full_directory):
        for (index, item) in enumerate(self._list):
            online_export(item, full_directory, "item_" + str(index))
    
    def load(self, directory, filename):
        if len(self._list) > 0: # avoid loading multiple times
            return False
        else:
            # Get full directory name
            full_directory = Folders.Folder(os.path.join(str(directory), filename))
            # Load list length
            len_ = self._load_len(full_directory)
            # Load list item vector dimension
            reference_vector = self._load_item_vector_dimension(full_directory)
            # Load list
            self._load_list(len_, reference_vector, full_directory)
            # Return
            return True
            
    def _load_len(self, full_directory):
        assert LenIO.exists_file(full_directory, "len")
        return LenIO.load_file(full_directory, "len")
        
    def _load_item_vector_dimension(self, full_directory):
        N = ItemVectorDimensionIO.load_file(full_directory, "item_vector_dimension")
        return OnlineVector(N)
        
    def _load_list(self, len_, reference_vector, full_directory):
        for index in range(len_):
            item = online_copy(reference_vector)
            online_import_(item, full_directory, "item_" + str(index))
            self._list.append(item)
            
    def __getitem__(self, key):
        return self._list[key]
        
    def __setitem__(self, key, item):
        self._list[key] = item
        
    def __iter__(self):
        return iter(self._list)
        
    def __len__(self):
        return len(self._list)
