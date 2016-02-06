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
        self.snapshots_matrix = SnapshotMatrix()
        self.eigensolver = OnlineEigenSolver()
        # 7. Inner product
        self.compute_scalar_product = compute_scalar_product_method
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
        self.snapshots_matrix.append(snapshot)
            
    ## Perform POD on the snapshots previously computed, and store the first
    #  POD modes in the basis functions matrix.
    #  Input arguments are: Nmax
    #  Output arguments are: POD modes, number of POD modes
    def apply(self, Nmax):
        dim = len(self.snapshot_matrix)
        correlation = np.matrix(np.zeros(dim, dim))
        for i in range(dim):
            for j in range(dim):
                correlation[i, j] = self.compute_scalar_product(self.snapshots_matrix[i], self.X, self.snapshots_matrix[j])
                
        eigensolver = OnlineEigenSolver(correlation, self.X)
        eigensolver.parameters["problem_type"] = "gen_hermitian"
        eigensolver.parameters["spectrum"] = "largest real"
        eigensolver.solve()

        Z = BasisFunctionsMatrix()
        for i in range(Nmax):
            print("lambda_" + str(i) + " = " + eigensolver.get_eigenvalue(i))
            eigv_i = eigensolver.get_eigenvector(i)
            Z_i = self.snapshots_matrix[0]*eigv_i[0]
            for j in range(1, dim):
                Z_i += self.snapshots_matrix[j]*eigv_i[j]
            Z_i /= self.compute_scalar_product(Z_i, self.X, Z_i)
            Z.enrich(Z_i)
        
        self.eigensolver = eigensolver
        return (Z, Nmax)

    def save_eigenvalues_file(self, output_directory, eigenvalues_file):
        self.eigensolver.save_eigenvalues_file(output_directory, eigenvalues_file)
        
    def save_retained_energy_file(self, output_directory, retained_energy_file):
        self.eigensolver.save_retained_energy_file(output_directory, retained_energy_file)        
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
