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

from rbnics.sampling import ParameterSpaceSubset
from rbnics.utils.decorators import overload

class GreedySelectedParametersList(object):
    def __init__(self):
        self.parameter_space_subset = ParameterSpaceSubset()
        
    def save(self, directory, filename):
        self.parameter_space_subset.save(directory, filename)
        
    def load(self, directory, filename):
        return self.parameter_space_subset.load(directory, filename)
        
    def append(self, element):
        self.parameter_space_subset.append(element)
        
    def closest(self, M, mu):
        output = GreedySelectedParametersList()
        output.parameter_space_subset = self.parameter_space_subset.closest(M, mu)
        return output
        
    @overload
    def __getitem__(self, key: int):
        return self.parameter_space_subset[key]
        
    @overload
    def __getitem__(self, key: slice):
        output = GreedySelectedParametersList()
        output.parameter_space_subset = self.parameter_space_subset[key]
        return output
        
    def __iter__(self):
        return iter(self.parameter_space_subset)
        
    def __len__(self):
        return len(self.parameter_space_subset)
        
    def __str__(self):
        return str(self.parameter_space_subset)
