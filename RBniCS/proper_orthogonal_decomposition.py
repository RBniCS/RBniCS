# Copyright (C) 2015-2016 SISSA mathLab
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

from dolfin import *
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~     PROPER ORTHOGONAL DECOMPOSITION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ProperOrthogonalDecomposition
#
# Class containing the implementation of the POD
class ProperOrthogonalDecomposition():

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of a POD object
    #  @{
    
    ## Default initialization of members
    def __init__(self):
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 6bis. Declare a matrix to store the snapshots
        self.snapshot_matrix = np.array([])
        
    ## Clean up
    def clear(self):
        self.snapshot_matrix = np.array([])
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
        
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Store a single snapshot in the snapshot matrix
    def store_single_snapshot(self, snapshot):
        if self.snapshot_matrix.size == 0: # for the first snapshot
            self.snapshot_matrix = np.array(snapshot.vector()).reshape(-1, 1) # as column vector
        else:
            self.snapshot_matrix = np.hstack((self.snapshot_matrix, np.array(snapshot.vector()).reshape(-1, 1))) # add new snapshots as column vectors
            
    ## Store a multiple snapshots in the snapshot matrix
    def store_multiple_snapshots(self, snapshots):
        if self.snapshot_matrix.size == 0: # for the first snapshot
            self.snapshot_matrix = np.array(snapshots) # as column vectors
        else:
            self.snapshot_matrix = np.hstack((self.snapshot_matrix, np.array(snapshots))) # add new snapshots as column vectors
            
    ## Perform POD on the snapshots previously computed, and store the first
    #  POD modes in the basis functions matrix.
    #  Input arguments are: inner product matrix, post processing file, Nmax and tol (mutually exclusive)
    #  Output arguments are: POD modes, number of POD modes
    def apply(self, S, eigenvalues_file, Nmax, tol):
        S = as_backend_type(S)
        dim = S.size(0) # = S.size(1)
        corr = np.matrix(np.dot(self.snapshot_matrix.T,np.matrix(np.dot(S.mat().getValues(range(dim),range(dim)),self.snapshot_matrix))))
        eigs, eigv = np.linalg.eig(corr)
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
        
        for i in range(Nmax):
            print "lambda_",i," = ",eigs[i]
            if i==0:
                p = np.dot(self.snapshot_matrix,eigv[:,i])
                p = np.squeeze(np.asarray(p)) # convert from an N_h x 1 matrix to an N_h vector
                p /= np.sqrt(np.dot(p, S*p))
                Z = p.reshape(-1, 1) # as column vector
            else:
                p = np.dot(self.snapshot_matrix,eigv[:,i])
                p = np.squeeze(np.asarray(p)) # convert from an N_h x 1 matrix to an N_h vector
                p /= np.sqrt(np.dot(p, S*p))
                Z = np.hstack((Z, p.reshape(-1, 1))) # add new basis functions as column vectors
        
        return (Z, Nmax)
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
