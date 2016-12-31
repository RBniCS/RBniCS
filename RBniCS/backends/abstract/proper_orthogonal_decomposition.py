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
## @file proper_orthogonal_decomposition.py
#  @brief Implementation of the POD
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators import AbstractBackend, abstractmethod

# Class containing the implementation of the POD
@AbstractBackend
class ProperOrthogonalDecomposition(object):

    ## Default initialization of members
    def __init__(self, V_or_Z, X, component=None):
        pass
        
    ## Clean up
    @abstractmethod
    def clear(self):
        pass
        
    ## Store a snapshot in the snapshot matrix
    @abstractmethod
    def store_snapshot(self, snapshot, component=None, weight=None):
        pass
            
    ## Perform POD on the snapshots previously computed, and store the first
    #  POD modes in the basis functions matrix.
    #  Input arguments are: Nmax
    #  Output arguments are: POD modes, number of POD modes
    @abstractmethod
    def apply(self, Nmax):
        pass

    @abstractmethod
    def print_eigenvalues(self, N=None):
        pass
        
    @abstractmethod
    def save_eigenvalues_file(self, output_directory, eigenvalues_file):
        pass
        
    @abstractmethod
    def save_retained_energy_file(self, output_directory, retained_energy_file):
        pass
    
