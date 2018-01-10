# Copyright (C) 2015-2018 by the RBniCS authors
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
from rbnics.backends.abstract import TensorsList as AbstractTensorsList
from rbnics.utils.decorators import overload
from rbnics.utils.mpi import is_io_process

def TensorsList(backend, wrapping, online_backend, online_wrapping):
    class _TensorsList(AbstractTensorsList):
        def __init__(self, space, empty_tensor):
            self.space = space
            self.empty_tensor = empty_tensor
            self.mpi_comm = wrapping.get_mpi_comm(space)
            self._list = list() # of tensors
            self._precomputed_slices = dict() # from tuple to TensorsList
        
        def enrich(self, tensors):
            # Append to storage
            self._enrich(tensors)
            # Reset precomputed slices
            self._precomputed_slices = dict()
            # Prepare trivial precomputed slice
            self._precomputed_slices[len(self._list)] = self
        
        @overload((backend.Matrix.Type(), backend.Vector.Type()), )
        def _enrich(self, tensors):
            self._list.append(wrapping.tensor_copy(tensors))
        
        @overload(lambda cls: cls, )
        def _enrich(self, tensors):
            for tensor in tensors:
                self._list.append(wrapping.tensor_copy(tensor))
            
        def clear(self):
            self._list = list()
            # Reset precomputed slices
            self._precomputed_slices = dict()
            
        def save(self, directory, filename):
            self._save_Nmax(directory, filename)
            for (index, tensor) in enumerate(self._list):
                wrapping.tensor_save(tensor, directory, filename + "_" + str(index))
                    
        def _save_Nmax(self, directory, filename):
            if is_io_process(self.mpi_comm):
                with open(os.path.join(str(directory), filename + ".length"), "w") as length:
                    length.write(str(len(self._list)))
            
        def load(self, directory, filename):
            if len(self._list) > 0: # avoid loading multiple times
                return False
            Nmax = self._load_Nmax(directory, filename)
            for index in range(Nmax):
                tensor = wrapping.tensor_copy(self.empty_tensor)
                loaded = wrapping.tensor_load(tensor, directory, filename + "_" + str(index))
                assert loaded
                self.enrich(tensor)
            return True
            
        def _load_Nmax(self, directory, filename):
            Nmax = None
            if is_io_process(self.mpi_comm):
                with open(os.path.join(str(directory), filename + ".length"), "r") as length:
                    Nmax = int(length.readline())
            Nmax = self.mpi_comm.bcast(Nmax, root=is_io_process.root)
            return Nmax
        
        @overload(online_backend.OnlineFunction.Type(), )
        def __mul__(self, other):
            return wrapping.tensors_list_mul_online_function(self, other)
        
        def __len__(self):
            return len(self._list)
        
        @overload(int)
        def __getitem__(self, key):
            return self._list[key]
            
        @overload(slice) # e.g. key = :N, return the first N tensors
        def __getitem__(self, key):
            assert key.start is None
            assert key.step is None
            assert key.stop <= len(self._list)
            
            if key.stop in self._precomputed_slices:
                return self._precomputed_slices[key.stop]
            else:
                output = _TensorsList.__new__(type(self), self.space, self.empty_tensor)
                output.__init__(self.space, self.empty_tensor)
                output._list = self._list[key]
                self._precomputed_slices[key.stop] = output
                return output
                
        def __iter__(self):
            return self._list.__iter__()
    return _TensorsList
