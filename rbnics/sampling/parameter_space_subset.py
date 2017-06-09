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

# Parameter space subsets
from rbnics.sampling.distributions import CompositeDistribution, UniformDistribution
from rbnics.utils.io import ExportableList
from rbnics.utils.mpi import is_io_process, parallel_max
from rbnics.utils.decorators import Extends, override
from numpy import zeros as array
from numpy import argmax

@Extends(ExportableList)
class ParameterSpaceSubset(ExportableList): # equivalent to a list of tuples
    @override
    def __init__(self, box):
        ExportableList.__init__(self, "pickle")
        self.box = box
        self.mpi_comm = is_io_process.mpi_comm # default communicator
        self.distributed_max = True
    
    # Method for generation of parameter space subsets
    def generate(self, n, sampling=None):
        if len(self.box) > 0:
            if is_io_process():
                if sampling is None:
                    sampling = UniformDistribution()
                elif isinstance(sampling, tuple):
                    assert len(sampling) == len(self.box)
                    sampling = CompositeDistribution(sampling)
                self._list = sampling.sample(self.box, n)
            self._list = is_io_process.mpi_comm.bcast(self._list, root=0)
        else:
            for i in range(n):
                self._list.append(tuple())
        
    @override
    def save(self, directory, filename):
        ExportableList.save(self, directory, filename)
        # Also save box
        self._FileIO.save_file(self.box, directory, filename + "_box")
        
    @override
    def load(self, directory, filename):
        result = ExportableList.load(self, directory, filename)
        if not result:
            return False
        # Also load the box
        assert self._FileIO.exists_file(directory, filename + "_box")
        box = self._FileIO.load_file(directory, filename + "_box")
        if len(box) != len(self.box):
            return False
        for (box_range, loaded_box_range) in zip(self.box, box):
            if box_range[0] != loaded_box_range[0] or box_range[1] != loaded_box_range[1]:
                return False
        return True
        
    def max(self, generator, postprocessor=None):
        if postprocessor is None:
            postprocessor = lambda value: value
        if self.distributed_max:
            local_list_indices = range(self.mpi_comm.rank, len(self._list), self.mpi_comm.size) # start from index rank and take steps of length equal to size
        else:
            local_list_indices = range(len(self._list))
        values = array(len(local_list_indices))
        values_with_postprocessing = array(len(local_list_indices))
        for i in range(len(local_list_indices)):
            values[i] = generator(self._list[ local_list_indices[i] ], local_list_indices[i])
            values_with_postprocessing[i] = postprocessor(values[i])
        if self.distributed_max:
            local_i_max = argmax(values_with_postprocessing)
            local_value_max = values[local_i_max]
            (global_value_max, global_i_max) = parallel_max(self.mpi_comm, local_value_max, local_list_indices[local_i_max], postprocessor)
            assert isinstance(global_i_max, tuple)
            assert len(global_i_max) == 1
            global_i_max = global_i_max[0]
        else:
            global_i_max = argmax(values_with_postprocessing)
            global_value_max = values[global_i_max]
        return (global_value_max, global_i_max)
    
    def diff(self, other_set):
        output = ParameterSpaceSubset(self.box)
        output.mpi_comm = self.mpi_comm
        output.distributed_max = self.distributed_max
        output._list = [mu for mu in self._list if mu not in other_set]
        return output
        
