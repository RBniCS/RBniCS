# Copyright (C) 2015 SISSA mathLab
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
## @file elliptic_coercive_pod_base.py
#  @brief Implementation of a POD-Galerkin ROM for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import scipy.linalg # for eigenvalue computation
import os # for path and makedir
import shutil # for rm
from elliptic_coercive_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE POD BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveBase
#
# Base class containing the interface of a POD-Galerkin ROM
# for elliptic coercive problems
class EllipticCoercivePODBase(EllipticCoerciveBase):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V):
    	# Call the parent initialization
        EllipticCoerciveBase.__init__(self, V)
        
    	# $$ OFFLINE DATA STRUCTURES $$ #
        # 6bis. Declare a new matrix to store the snapshots
        self.snapshot_matrix = np.array([])
        # 9. I/O
        self.snap_folder = "snapshots__pod/"
        self.basis_folder = "basis__pod/"
        self.dual_folder = "dual__pod/" # never used
        self.rb_matrices_folder = "red_matr__pod/"
        self.pp_folder = "pp__pod/" # post processing
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Nothing to be added in this case
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
    def offline(self):
        print "=============================================================="
        print "=             Offline phase begins                           ="
        print "=============================================================="
        print ""
        if os.path.exists(self.pp_folder):
            shutil.rmtree(self.pp_folder)
        folders = (self.snap_folder, self.basis_folder, self.red_matrices_folder, self.pp_folder)
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)
        
        self.truth_A = self.assemble_truth_a()
        self.truth_F = self.assemble_truth_f()
        self.Qa = len(self.truth_A)
        self.Qf = len(self.truth_F)
        
        run = 0
        for mu in self.xi_train:
            print "############################## run = ", run, " ######################################"
            
            self.setmu(self.xi_train[run])
            self.theta_a = self.compute_theta_a()
            self.theta_f = self.compute_theta_f()
            
            print "truth solve for mu = ", self.mu
            self.truth_solve()
            
            print "update snapshot matrix"
            self.update_snapshot_matrix()

            print ""
            run += 1
            
        print "############################## perform POD ######################################"
        self.N = self.pod()
        
        print ""
        print "build reduced matrices"
        self.build_red_matrices()
        self.build_red_vectors()
        np.save(self.red_matrices_folder + "red_A", self.red_A)
        np.save(self.red_matrices_folder + "red_F", self.red_F)
        
        print ""
        print "=============================================================="
        print "=             Offline phase ends                             ="
        print "=============================================================="
        print ""

    ## Update the snapshot matrix
    def update_snapshot_matrix(self):
        if self.snapshot_matrix.size == 0: # for the first snapshot
            self.snapshot_matrix = np.array(self.snap.vector()).reshape(-1, 1) # as column vector
        else:
            self.snapshot_matrix = np.hstack((self.snapshot_matrix, self.snap.vector())) # add new snapshots as column vectors
            
    ## Perform POD on the snapshots previously computed, and store the first
    #  POD modes in the basis functions matrix
    def pod(self):
        dim = self.dim
        S = self.S
        S = as_backend_type(S)
        corr = np.matrix(np.dot(self.snapshot_matrix.T,np.matrix(np.dot(S.mat().getValues(range(dim),range(dim)),self.snapshot_matrix))))
        eigs, eigv = scipy.linalg.eig(corr)
        idx = eigs.argsort()
        idx = idx[::-1]
        eigs = eigs[idx]
   
        eigv = eigv[:,idx]
        np.save(self.pp_folder + "eigs",eigs)
        
        # Remove (negigible) complex parts
        eigs = np.real(eigs)
        eigv = np.real(eigv)
        
        tot = np.sum(eigs)
        eigs_norm = eigs/tot
        
        for i in range(self.Nmax):
            print "lambda_",i," = ",eigs[i]
            if i==0:
                p = np.dot(self.snapshot_matrix,eigv[:,i])
                p /= np.sqrt(np.dot(p, self.S*p))
                self.Z = p.reshape(-1, 1) # as column vector
            else:
                p = np.dot(self.snapshot_matrix,eigv[:,i])
                p /= np.sqrt(np.dot(p, self.S*p))
                self.Z = np.hstack((self.Z, p.reshape(-1, 1))) # add new basis functions as column vectors
        
        return self.Nmax
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    # Nothing to be added in this case
    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
