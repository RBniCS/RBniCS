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

import os # for path and makedir
import shutil # for rm
from scipy import stats as scistats
from proper_orthogonal_decomposition import *
from elliptic_coercive_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE POD BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoercivePODBase
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
        # 6bis. Declare a POD object
        self.POD = ProperOrthogonalDecomposition()
        # 9. I/O
        self.snap_folder = "snapshots__pod/"
        self.basis_folder = "basis__pod/"
        self.dual_folder = "dual__pod/" # never used
        self.red_matrices_folder = "red_matr__pod/"
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
        
        for run in range(len(self.xi_train)):
            print "############################## run = ", run, " ######################################"
            
            self.setmu(self.xi_train[run])
            
            print "truth solve for mu = ", self.mu
            self.truth_solve()
            self.export_solution(self.snap, self.snap_folder + "truth_" + str(run))
            
            print "update snapshot matrix"
            self.update_snapshot_matrix()

            print ""
            run += 1
            
        print "############################## perform POD ######################################"
        (self.Z, self.N) = self.apply_POD(self.S, self.pp_folder + "eigs", self.Nmax, self.tol)
        
        print ""
        print "build reduced matrices"
        self.build_red_matrices()
        self.build_red_vectors()
        
        print ""
        print "=============================================================="
        print "=             Offline phase ends                             ="
        print "=============================================================="
        print ""
        
    ## Update the snapshot matrix
    def update_snapshot_matrix(self):
        self.POD.store_single_snapshot(self.snap)
        
    ## Apply POD
    def apply_POD(self, S, pp_file, Nmax, tol):
        (Z, N) = self.POD.apply(S, pp_file, Nmax, tol)
        np.save(self.basis_folder + "basis", self.Z)
        current_basis = Function(self.V)
        for b in range(N):
            current_basis.vector()[:] = Z[:, b]
            self.export_solution(current_basis, self.basis_folder + "basis_" + str(b))
        return (Z, N)
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the training set
    def error_analysis(self, N=None):
        if N is None:
            N = self.N
            
        print "=============================================================="
        print "=             Error analysis begins                          ="
        print "=============================================================="
        print ""
        
        # Generate a new test set
        self.setxi_train(len(self.xi_train))
        
        error = np.zeros((N, len(self.xi_train)))
        
        for run in range(len(self.xi_train)):
            print "############################## run = ", run, " ######################################"
            
            self.setmu(self.xi_train[run])
            
            # Perform the truth solve only once
            self.truth_solve()
            
            for n in range(N): # n = 0, 1, ... N - 1
                error[n, run] = self.compute_error(n + 1, False)
        
        # Print some statistics
        print ""
        print "N \t gmean(err)"
        for n in range(N): # n = 0, 1, ... N - 1
            mean_error = scistats.gmean(error[n, :])
            print str(n+1) + " \t " + str(mean_error)
        
        print ""
        print "=============================================================="
        print "=             Error analysis ends                            ="
        print "=============================================================="
        print ""
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    # Nothing to be added in this case
    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
