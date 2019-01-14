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

import operator # to find closest parameters
from math import sqrt
from mpi4py.MPI import COMM_WORLD
from numpy import zeros as array
from numpy import argmax
from rbnics.sampling.distributions import CompositeDistribution, UniformDistribution
from rbnics.utils.decorators import overload
from rbnics.utils.io import ExportableList
from rbnics.utils.mpi import parallel_io as parallel_generate, parallel_max

class ParameterSpaceSubset(ExportableList): # equivalent to a list of tuples
    def __init__(self):
        ExportableList.__init__(self, "text")
        self.mpi_comm = COMM_WORLD
        self.distributed_max = True
        
    @overload
    def __getitem__(self, key: int):
        return self._list[key]
        
    @overload
    def __getitem__(self, key: slice):
        output = ParameterSpaceSubset()
        output.distributed_max = self.distributed_max
        output._list = self._list[key]
        return output
    
    # Method for generation of parameter space subsets
    def generate(self, box, n, sampling=None):
        if len(box) > 0:
            if sampling is None:
                sampling = UniformDistribution()
            elif isinstance(sampling, tuple):
                assert len(sampling) == len(box)
                sampling = CompositeDistribution(sampling)
            def run_sampling():
                return sampling.sample(box, n)
            self._list = parallel_generate(run_sampling, self.mpi_comm)
        else:
            for i in range(n):
                self._list.append(tuple())
        
    def max(self, generator, postprocessor=None):
        if postprocessor is None:
            def postprocessor(value):
                return value
        if self.distributed_max:
            local_list_indices = list(range(self.mpi_comm.rank, len(self._list), self.mpi_comm.size)) # start from index rank and take steps of length equal to size
        else:
            local_list_indices = list(range(len(self._list)))
        values = array(len(local_list_indices))
        values_with_postprocessing = array(len(local_list_indices))
        for i in range(len(local_list_indices)):
            values[i] = generator(self._list[local_list_indices[i]])
            values_with_postprocessing[i] = postprocessor(values[i])
        if self.distributed_max:
            local_i_max = argmax(values_with_postprocessing)
            local_value_max = values[local_i_max]
            (global_value_max, global_i_max) = parallel_max(local_value_max, local_list_indices[local_i_max], postprocessor, self.mpi_comm)
            assert isinstance(global_i_max, tuple)
            assert len(global_i_max) == 1
            global_i_max = global_i_max[0]
        else:
            global_i_max = argmax(values_with_postprocessing)
            global_value_max = values[global_i_max]
        return (global_value_max, global_i_max)
        
    def serialize_maximum_computations(self):
        assert self.distributed_max is True
        self.distributed_max = False
    
    def diff(self, other_set):
        output = ParameterSpaceSubset()
        output.distributed_max = self.distributed_max
        output._list = [mu for mu in self._list if mu not in other_set]
        return output
        
    # M parameters in this set closest to mu
    def closest(self, M, mu):
        assert M <= len(self)
        
        # Trivial case 1:
        if M == len(self):
            return self
            
        output = ParameterSpaceSubset()
        output.distributed_max = self.distributed_max
            
        # Trivial case 2:
        if M == 0:
            return output
        
        parameters_and_distances = list()
        for xi_i in self:
            distance = sqrt(sum([(x - y)**2 for (x, y) in zip(mu, xi_i)]))
            parameters_and_distances.append((xi_i, distance))
        parameters_and_distances.sort(key=operator.itemgetter(1))
        output._list = [xi_i for (xi_i, _) in parameters_and_distances[:M]]
        return output
