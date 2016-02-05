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

from __future__ import print_function
from config import *
from dolfin import *
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~     PROPER ORTHOGONAL DECOMPOSITION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ProperOrthogonalDecomposition
#
# Class containing the implementation of the POD
class ProperOrthogonalDecomposition(object):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of a POD object
    #  @{
    
    ## Default initialization of members
    def __init__(self, compute_scalar_product_method, X):
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 6bis. Declare a matrix to store the snapshots
        self.snapshot_matrix = []
        # 7. Inner product
        self.compute_scalar_product = compute_scalar_product_method
        self.X = X
        
    ## Clean up
    def clear(self):
        self.snapshot_matrix = []
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
        
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Store a single snapshot in the snapshot matrix
    def store_single_snapshot(self, snapshot):
        self.snapshot_matrix.append(snapshot.vector().copy())
            
    ## Store a multiple snapshots in the snapshot matrix
    def store_multiple_snapshots(self, snapshots):
        self.snapshot_matrix.extend(snapshots) # note the difference between extend and append in python
            
    ## Perform POD on the snapshots previously computed, and store the first
    #  POD modes in the basis functions matrix.
    #  Input arguments are: post processing file, Nmax
    #  Output arguments are: POD modes, number of POD modes
    def apply(self, eigenvalues_file, Nmax):
        dim = len(self.snapshot_matrix)
        correlation = np.matrix(np.zeros(dim, dim))
        for i in range(dim):
            for j in range(dim):
                correlation[i, j] = self.compute_scalar_product(self.snapshot_matrix[i], self.X, self.snapshot_matrix[j])
        eigs, eigv = np.linalg.eig(correlation)
        idx = eigs.argsort()
        idx = idx[::-1]
        eigs = eigs[idx]
   
        eigv = eigv[:,idx]
        np.save(eigenvalues_file,eigs)
        
        # Remove (negigible) complex parts
        eigs = np.real(eigs)
        eigv = np.real(eigv)
        
        tot = np.sum(eigs)
        eigs_norm = eigs/tot
        
        Z = []
        for i in range(Nmax):
            print("lambda_",i," = ",eigs[i])
            Z_i = self.snapshot_matrix[0]*eigv[0, i]
            for j in range(1, dim):
                Z_i += self.snapshot_matrix[j]*eigv[j, i]
            Z_i /= self.compute_scalar_product(Z_i, self.X, Z_i)
            Z.append(Z_i.copy())
        
        return (Z, Nmax)
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
