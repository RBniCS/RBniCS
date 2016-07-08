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
## @file proper_orthogonal_decomposition.py
#  @brief Implementation of the POD
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.linear_algebra.transpose import transpose
from RBniCS.linear_algebra.snapshots_matrix import SnapshotsMatrix
from RBniCS.linear_algebra.online_eigen_solver import OnlineEigenSolver

#~~~~~~~~~~~~~~~~~~~~~~~~~     PROPER ORTHOGONAL DECOMPOSITION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ProperOrthogonalDecomposition
#
# Class containing the implementation of the POD
class ProperOrthogonalDecomposition(object):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of a POD object
    #  @{
    
    ## Default initialization of members
    def __init__(self, X):
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Declare a matrix to store the snapshots
        self.snapshots_matrix = SnapshotsMatrix()
        self.eigensolver = OnlineEigenSolver()
        # Inner product
        self.X = X
        
    ## Clean up
    def clear(self):
        self.snapshots_matrix = SnapshotMatrix()
        self.eigensolver = OnlineEigenSolver()
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
        
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Store a snapshot in the snapshot matrix
    def store_snapshot(self, snapshot):
        self.snapshots_matrix.enrich(snapshot)
            
    ## Perform POD on the snapshots previously computed, and store the first
    #  POD modes in the basis functions matrix.
    #  Input arguments are: Nmax
    #  Output arguments are: POD modes, number of POD modes
    def apply(self, Nmax):
        assert len(self.X) == 1 # note that we cannot move this assert in __init__ because
                                # self.X has not been assembled yet there
        X = self.X[0]
        snapshots_matrix = self.snapshots_matrix
        
        correlation = transpose(snapshots_matrix)*X*snapshots_matrix
        
        eigensolver = OnlineEigenSolver(correlation)
        eigensolver.parameters["problem_type"] = "hermitian"
        eigensolver.parameters["spectrum"] = "largest real"
        eigensolver.solve()
        
        Z = snapshots_matrix*eigensolver.get_eigenvectors(Nmax)
        for b in Z:
            b /= transpose(b)*X*b
        
        self.eigensolver = eigensolver
        return (Z, Nmax)

    def print_eigenvalues(self, N=None):
        self.eigensolver.print_eigenvalues(N)
        
    def save_eigenvalues_file(self, output_directory, eigenvalues_file):
        self.eigensolver.save_eigenvalues_file(output_directory, eigenvalues_file)
        
    def save_retained_energy_file(self, output_directory, retained_energy_file):
        self.eigensolver.save_retained_energy_file(output_directory, retained_energy_file)        
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
