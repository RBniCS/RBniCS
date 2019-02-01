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
from rbnics.backends.online import OnlineAffineExpansionStorage
from rbnics.utils.io import Folders, TextIO as NmaxIO
from rbnics.utils.decorators import overload, tuple_of

class OnlineNonHierarchicalAffineExpansionStorage(object):
    def __init__(self, arg1):
        self._content = dict()
        self._len = arg1
        
    @overload(slice)
    def __getitem__(self, key):
        N = self._convert_key(key)
        assert N in self._content
        return self._content[N]
        
    @overload(tuple_of(slice))
    def __getitem__(self, key):
        assert len(key) == 2
        assert key[0] == key[1]
        return self.__getitem__(key[0])
        
    @overload(slice, OnlineAffineExpansionStorage)
    def __setitem__(self, key, item):
        N = self._convert_key(key)
        assert len(item) == self._len
        self._content[N] = item
        
    @overload(tuple_of(slice), OnlineAffineExpansionStorage)
    def __setitem__(self, key, item):
        assert len(key) == 2
        assert key[0] == key[1]
        return self.__setitem__(key[0], item)
        
    def _convert_key(self, key):
        assert key.start is None
        assert key.step is None
        assert isinstance(key.stop, (dict, int))
        if isinstance(key.stop, dict):
            assert len(key.stop) == 1
            assert "u" in key.stop
            N = key.stop["u"]
        else:
            N = key.stop
        return N
        
    def save(self, directory, filename):
        # Get full directory name
        full_directory = Folders.Folder(os.path.join(str(directory), filename))
        full_directory.create()
        # Save Nmax
        self._save_Nmax(full_directory)
        # Save non hierarchical content
        for (N, affine_expansion_N) in self._content.items():
            self._save_content(N, affine_expansion_N, directory, filename)
            
    def _save_Nmax(self, full_directory):
        if len(self._content) > 0:
            assert min(self._content.keys()) == 1
            assert max(self._content.keys()) == len(self._content)
        NmaxIO.save_file(len(self._content), full_directory, "Nmax")
            
    def _save_content(self, N, affine_expansion_N, directory, filename):
        affine_expansion_N.save(directory, filename + "_N=" + str(N))
        
    def load(self, directory, filename):
        if len(self._content) > 0: # avoid loading multiple times
            return False
        # Get full directory name
        full_directory = Folders.Folder(os.path.join(str(directory), filename))
        # Load Nmax
        Nmax = self._load_Nmax(full_directory)
        # Load non hierarchical content
        for N in range(1, Nmax + 1):
            self._content[N] = self._load_content(N, directory, filename)
        # Return
        return True
        
    def _load_Nmax(self, full_directory):
        assert NmaxIO.exists_file(full_directory, "Nmax")
        return NmaxIO.load_file(full_directory, "Nmax")
        
    def _load_content(self, N, directory, filename):
        affine_expansion_N = OnlineAffineExpansionStorage(self._len)
        loaded = affine_expansion_N.load(directory, filename + "_N=" + str(N))
        assert loaded
        return affine_expansion_N
        
    def __len__(self):
        return self._len
