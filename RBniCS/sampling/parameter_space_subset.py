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
## @file parameter_space_subset.py
#  @brief Type for parameter space subsets
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

# Parameter space subsets
import itertools # for linspace sampling
import numpy
from RBniCS.io_utils import ExportableList
from RBniCS.sampling.distributions import UniformDistribution

class ParameterSpaceSubset(ExportableList): # equivalent to a list of tuples
    def __init__(self, box):
        ExportableList.__init__(self, "pickle")
        self.box = box
    
    # Method for generation of parameter space subsets
    def generate(self, n, sampling):
        if sampling == None:
            sampling = UniformDistribution()
        self._list = sampling.sample(self.box, n)
        
    def load(self, directory, filename):
        result = ExportableList.load(self, directory, filename)
        if not result:
            return False
        # Also load the box
        assert self._FileIO.exists_file(directory, filename + "_box")
        box = self._FileIO.load_file(directory, filename + "_box")
        if len(box) != len(self.box):
            return False
        for p in range(len(box)):
            if box[p][0] != self.box[p][0] or box[p][1] != self.box[p][1]:
                return False
        return True
        
        
    def save(self, directory, filename):
        ExportableList.save(self, directory, filename)
        # Also save box
        self._FileIO.save_file(self.box, directory, filename + "_box")
        
#  @}
########################### end - OFFLINE STAGE - end ########################### 

