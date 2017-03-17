# Copyright (C) 2015-2017 by the RBniCS authors
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
## @file tensors_list.py
#  @brief Type for storing a list of FE tensors.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends.abstract import TensorsList as AbstractTensorsList
from RBniCS.utils.decorators import Extends, override
from RBniCS.utils.mpi import is_io_process

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

@Extends(AbstractTensorsList)
class TensorsList(AbstractTensorsList):
    @override
    def __init__(self, V_or_Z, empty_tensor, backend, wrapping, online_backend):
        self.V_or_Z = V_or_Z
        self.empty_tensor = empty_tensor
        self.mpi_comm = wrapping.get_mpi_comm(V_or_Z)
        self.backend = backend
        self.wrapping = wrapping
        self.online_backend = online_backend
        self._list = list() # of tensors
        self._precomputed_slices = dict() # from tuple to TensorsList
    
    @override
    def enrich(self, tensors):
        # Append to storage
        assert isinstance(tensors, (TensorsList, self.backend.Matrix.Type(), self.backend.Vector.Type()))
        if isinstance(tensors, TensorsList):
            for tensor in tensors:
                self._list.append(self.wrapping.tensor_copy(tensor))
        elif isinstance(tensors, (self.backend.Matrix.Type(), self.backend.Vector.Type())):
            self._list.append(self.wrapping.tensor_copy(tensors))
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in TensorsList.enrich.")
        # Reset precomputed slices
        self._precomputed_slices = dict()
        # Prepare trivial precomputed slice
        self._precomputed_slices[len(self._list)] = self
        
    @override
    def clear(self):
        self._list = list()
        # Reset precomputed slices
        self._precomputed_slices = dict()
        
    @override
    def save(self, directory, filename):
        self._save_Nmax(directory, filename)
        for (index, tensor) in enumerate(self._list):
            self.wrapping.tensor_save(tensor, directory, filename + "_" + str(index))
                
    def _save_Nmax(self, directory, filename):
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + ".length", "w") as length:
                length.write(str(len(self._list)))
        
    @override
    def load(self, directory, filename):
        if len(self._list) > 0: # avoid loading multiple times
            return False
        Nmax = self._load_Nmax(directory, filename)
        for index in range(Nmax):
            tensor = backend.copy(self.empty_tensor)
            self.wrapping.tensor_load(tensor, directory, filename + "_" + str(index))
            self.enrich(tensor)
        return True
        
    def _load_Nmax(self, directory, filename):
        Nmax = None
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + ".length", "r") as length:
                Nmax = int(length.readline())
        Nmax = self.mpi_comm.bcast(Nmax, root=is_io_process.root)
        return Nmax
    
    @override
    def __mul__(self, other):
        assert isinstance(other, self.online_backend.Function.Type())
        return self.wrapping.tensors_list_mul_online_function(self, other)
    
    @override
    def __len__(self):
        return len(self._list)

    @override
    def __getitem__(self, key):
        if isinstance(key, slice): # e.g. key = :N, return the first N tensors
            assert key.start is None 
            assert key.step is None
            assert key.stop <= len(self._list)
            
            if key.stop in self._precomputed_slices:
                return self._precomputed_slices[key.stop]
            else:
                output = self.backend.TensorsList(self.V_or_Z)
                output._list = self._list[key]
                self._precomputed_slices[key.stop] = output
                return output
                
        else: # return the element at position "key" in the storage
            return self._list[key]
            
    @override
    def __iter__(self):
        return self._list.__iter__()
        
