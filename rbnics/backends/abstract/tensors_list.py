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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators import AbstractBackend, abstractmethod

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

@AbstractBackend
class TensorsList(object):
    def __init__(self, V_or_Z, empty_tensor):
        pass
    
    @abstractmethod
    def enrich(self, tensor):
        pass
        
    @abstractmethod
    def clear(self):
        pass
        
    @abstractmethod
    def save(self, directory, filename):
        pass
        
    @abstractmethod
    def load(self, directory, filename):
        pass
        
    # self * other [used to compute S*eigv]
    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __len__(self):
        pass
            
    # key may be an integer
    @abstractmethod
    def __getitem__(self, key):
        pass
        
    @abstractmethod
    def __iter__(self):
        pass
        
