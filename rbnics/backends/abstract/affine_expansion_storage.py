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
## @file affine_expansion_online_storage.py
#  @brief Type for storing online quantities related to an affine expansion
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators import AbstractBackend, abstractmethod, abstractonlinemethod

@AbstractBackend
class AffineExpansionStorage(object):
    def __init__(self):
        pass
        
    @abstractonlinemethod
    def save(self, directory, filename):
        pass
        
    @abstractonlinemethod
    def load(self, directory, filename):
        pass
    
    @abstractmethod
    def __getitem__(self, key):
        pass
        
    @abstractmethod
    def __iter__(self):
        pass
        
    @abstractonlinemethod
    def __setitem__(self, key, item):
        pass
        
    @abstractmethod
    def __len__(self):
        pass
        
