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
## @file elliptic_coercive_rb_base.py
#  @brief Implementation of the reduced basis method for (compliant) elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import os # for path and makedir
import shutil # for rm
import sys # for exit
import random # to randomize selection in case of equal error bound
from gram_schmidt import *
from elliptic_coercive_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE RB BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveRBBase
#
# Base class containing the interface of the RB method
# for (compliant) elliptic coercive problems
class EllipticCoerciveRBBase(EllipticCoerciveBase):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, bc_list):
        # Call the parent initialization
        EllipticCoerciveBase.__init__(self, V, bc_list)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # 4. Online output
        self.sN = 0
        # 5. Residual terms
        self.Cf = ()
        self.lnq = ()
        self.CC = np.array([])
        self.CL = np.array([])
        self.LL = np.array([])
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 4. Offline output
        self.s = 0
        # 6bis. Declare a GS object
        self.GS = GramSchmidt()
        # 9. I/O
        self.snapshots_folder = "snapshots/"
        self.basis_folder = "basis/"
        self.dual_folder = "dual/"
        self.reduced_matrices_folder = "reduced_matrices/"
        self.post_processing_folder = "post_processing/"
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Perform an online evaluation of the (compliant) output
    def online_output(self):
        N = self.uN.size
        self.theta_f = self.compute_theta_f()
        assembled_reduced_F = self.affine_assemble_reduced_vector(self.reduced_F, self.theta_f, N)
        self.sN = float(np.dot(assembled_reduced_F, self.uN))
        
    ## Return an error bound for the current solution
    def get_delta(self):
        eps2 = self.get_eps2()
        alpha = self.get_alpha_lb()
        return np.sqrt(np.abs(eps2)/alpha)
    
    ## Return an error bound for the current output
    def get_delta_output(self):
        eps2 = self.get_eps2()
        alpha = self.get_alpha_lb()
        return np.abs(eps2)/alpha
        
    ## Return the numerator of the error bound for the current solution
    def get_eps2 (self):
        theta_a = self.theta_a
        theta_f = self.theta_f
        Qf = self.Qf
        Qa = self.Qa
        uN = self.uN
        
        eps2 = 0.0
        
        CC = self.CC
        for qf in range(Qf):
            for qfp in range(Qf):
                eps2 += theta_f[qf]*theta_f[qfp]*CC[qf,qfp]
        
        CL = self.CL
        LL = self.LL
        if self.N == 1:
            for qf in range(Qf):
                for qa in range(Qa):
                    eps2 += 2.0*theta_f[qf]*uN*theta_a[qa]*CL[0,qf,qa]
    
    
            for qa in range(Qa):
                for qap in range(Qa):
                    eps2 += theta_a[qa]*uN*uN*theta_a[qap]*LL[0,0,qa,qap]
            
        else:
            n = 0
            for un in uN:
                for qf in range(Qf):
                    for qa in range(Qa):
                        eps2 += 2.0*theta_f[qf]*theta_a[qa]*un*CL[n,qf,qa]
                n += 1

            n = 0
            for un in uN:
                for qa in range(Qa):
                    np = 0
                    for unp in uN:
                        for qap in range(Qa):
                            eps2 += theta_a[qa]*un*theta_a[qap]*unp*LL[n,np,qa,qap]
                        np += 1
                n += 1
        
        return eps2
        
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
        if os.path.exists(self.post_processing_folder):
            shutil.rmtree(self.post_processing_folder)
        folders = (self.snapshots_folder, self.basis_folder, self.dual_folder, self.reduced_matrices_folder, self.post_processing_folder)
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)
        
        self.truth_A = self.assemble_truth_a()
        self.apply_bc_to_matrix_expansion(self.truth_A)
        self.truth_F = self.assemble_truth_f()
        self.apply_bc_to_vector_expansion(self.truth_F)
        self.Qa = len(self.truth_A)
        self.Qf = len(self.truth_F)
        
        for run in range(self.Nmax):
            print "############################## run = ", run, " ######################################"
            
            print "truth solve for mu = ", self.mu
            self.truth_solve()
            self.export_solution(self.snapshot, self.snapshots_folder + "truth_" + str(run))
            
            print "update basis matrix"
            self.update_basis_matrix()
            
            print "build reduced matrices"
            self.build_reduced_matrices()
            self.build_reduced_vectors()
            
            print "reduced order solve"
            self._online_solve(self.N)
            
            print "build matrices for error estimation (it may take a while)"
            self.compute_dual_terms()
            
            if self.N < self.Nmax:
                print "find next mu"
                self.greedy()
            else:
                self.greedy()

            print ""
            
        print "=============================================================="
        print "=             Offline phase ends                             ="
        print "=============================================================="
        print ""
        
    ## Update basis matrix
    def update_basis_matrix(self):
        if self.N == 0:
            self.Z = np.array(self.snapshot.vector()).reshape(-1, 1) # as column vector
            self.Z /= np.sqrt(np.dot(self.Z[:, 0], self.S*self.Z[:, 0]))
        else:
            self.Z = np.hstack((self.Z, np.array(self.snapshot.vector()).reshape(-1, 1))) # add new basis functions as column vectors
            self.Z = self.GS.apply(self.Z, self.S)
        np.save(self.basis_folder + "basis", self.Z)
        current_basis = Function(self.V)
        current_basis.vector()[:] = np.array(self.Z[:, self.N], dtype=np.float_)
        self.export_basis(current_basis, self.basis_folder + "basis_" + str(self.N))
        self.N += 1
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        delta_max = -1.0
        munew = None
        for mu in self.xi_train:
            self.setmu(mu)
            self._online_solve(self.N)
            delta = self.get_delta()
            if (delta > delta_max or (delta == delta_max and random.random() >= 0.5)):
                delta_max = delta
                munew = mu
        print "absolute delta max = ", delta_max
        if os.path.isfile(self.post_processing_folder + "delta_max.npy") == True:
            d = np.load(self.post_processing_folder + "delta_max.npy")
            
            np.save(self.post_processing_folder + "delta_max", np.append(d, delta_max))
    
            m = np.load(self.post_processing_folder + "mu_greedy.npy")
            np.save(self.post_processing_folder + "mu_greedy", np.append(m, munew))
        else:
            np.save(self.post_processing_folder + "delta_max", delta_max)
            np.save(self.post_processing_folder + "mu_greedy", np.array(munew))

        self.setmu(munew)
        
    ## Compute dual terms
    def compute_dual_terms(self):
        N = self.N
        RBu = Function(self.V)
        
        Qf = self.Qf
        Qa = self.Qa
        if N == 1 :
            
            # CC (does not depend on N, so we compute it only once)
            self.Cf = self.compute_f_dual()
            self.CC = np.zeros((Qf,Qf))
            for qf in range(0,Qf):
                for qfp in range(qf,Qf):
                    self.CC[qf,qfp] = self.compute_scalar(self.Cf[qf],self.Cf[qfp],self.S)
                    if qf != qfp:
                        self.CC[qfp,qf] = self.CC[qf,qfp]
            np.save(self.dual_folder + "CC", self.CC)
    
            RBu.vector()[:] = self.Z[:, 0]
            
            self.lnq = (self.compute_a_dual(RBu),)
    
            la = Function(self.V)
            lap = Function(self.V)

            # CL
            self.CL = np.zeros((self.Nmax,Qf,Qa))
            for qf in range(0,Qf):
                for qa in range(0,Qa):
                    la.vector()[:] = np.array(self.lnq[0][:,qa], dtype=np.float_)
                    self.CL[0,qf,qa] = self.compute_scalar(la,self.Cf[qf],self.S)
            np.save(self.dual_folder + "CL", self.CL)
            
            # LL
            self.LL = np.zeros((self.Nmax,self.Nmax,Qa,Qa))
            for qa in range(0,Qa):
                la.vector()[:] = np.array(self.lnq[0][:,qa], dtype=np.float_)
                for qap in range(qa,Qa):
                    lap.vector()[:] = np.array(self.lnq[0][:,qap], dtype=np.float_)
                    self.LL[0,0,qa,qap] = self.compute_scalar(la,lap,self.S)
                    if qa != qap:
                        self.LL[0,0,qap,qa] = self.LL[0,0,qa,qap]
            np.save(self.dual_folder + "LL", self.LL)
        else:
            RBu.vector()[:] = np.array(self.Z[:, N-1], dtype=np.float_)
            self.lnq += (self.compute_a_dual(RBu),)
            la = Function(self.V)
            lap = Function(self.V)
            cl = np.zeros((Qf,Qa))
            n = self.N-1

            # CL
            for qf in range(0,Qf):
                for qa in range(0,Qa):
                    la.vector()[:] = np.array(self.lnq[n][:, qa], dtype=np.float_)
                    self.CL[n,qf,qa] = self.compute_scalar(self.Cf[qf],la,self.S)
            np.save(self.dual_folder + "CL", self.CL)
    
            # LL
            for qa in range(0,Qa):
                la.vector()[:] = np.array(self.lnq[n][:, qa], dtype=np.float_)
                for nn in range(0,N):
                    for qap in range(0,Qa):
                        lap.vector()[:] = np.array(self.lnq[nn][:, qap], dtype=np.float_)
                        self.LL[n,nn,qa,qap] = self.compute_scalar(la,lap,self.S)
                        if n != nn:
                            self.LL[nn,n,qa,qap] = self.LL[n,nn,qa,qap]
            np.save(self.dual_folder + "LL", self.LL)
    
    ## Compute the dual of a
    def compute_a_dual(self, RBu):
        riesz = Function(self.V)
        i = 0
        for A in self.truth_A:
            solve (self.S, riesz.vector(), A*RBu.vector()*(-1.0))
            if i != 0:
                l = np.hstack((l,np.array(riesz.vector()).reshape(-1, 1)))
            else:
                l = np.array(riesz.vector()).reshape(-1, 1) # as column vector
                i = 1
        return l
    
    ## Compute the dual of f
    def compute_f_dual(self):
        riesz = Function(self.V)
        c = ()
        for F in self.truth_F:
            solve (self.S, riesz.vector(), F)
            c += (riesz.copy(True),)
        return c
        
    ## Perform a truth evaluation of the output
    def truth_output(self):
        self.theta_f = self.compute_theta_f()
        assembled_truth_F = self.affine_assemble_truth_vector(self.truth_F, self.theta_f)
        self.s = assembled_truth_F.inner(self.snapshot.vector())
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu. Overridden to compute also error on the output
    def compute_error(self, N=None, skip_truth_solve=False):
        error_u = EllipticCoerciveBase.compute_error(self, N, skip_truth_solve)
        if not skip_truth_solve:
            self.truth_output()
        self.online_output()
        error_s = abs(self.s - self.sN)
        return (error_u, error_s)
        
        
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    def error_analysis(self, N=None):
        self.load_reduced_matrices()
        if N is None:
            N = self.N
            
        self.truth_A = self.assemble_truth_a()
        self.apply_bc_to_matrix_expansion(self.truth_A)
        self.truth_F = self.assemble_truth_f()
        self.apply_bc_to_vector_expansion(self.truth_F)
        self.Qa = len(self.truth_A)
        self.Qf = len(self.truth_F)
        
        print "=============================================================="
        print "=             Error analysis begins                          ="
        print "=============================================================="
        print ""
        
        error_u = np.zeros((N, len(self.xi_test)))
        delta_u = np.zeros((N, len(self.xi_test)))
        effectivity_u = np.zeros((N, len(self.xi_test)))
        error_s = np.zeros((N, len(self.xi_test)))
        delta_s = np.zeros((N, len(self.xi_test)))
        effectivity_s = np.zeros((N, len(self.xi_test)))
        
        for run in range(len(self.xi_test)):
            print "############################## run = ", run, " ######################################"
            
            self.setmu(self.xi_test[run])
            
            # Perform the truth solve only once
            self.truth_solve()
            self.truth_output()
            
            for n in range(N): # n = 0, 1, ... N - 1
                (current_error_u, current_error_s) = self.compute_error(n + 1, True)
                
                error_u[n, run] = current_error_u
                delta_u[n, run] = self.get_delta()
                effectivity_u[n, run] = delta_u[n, run]/error_u[n, run]
                
                error_s[n, run] = current_error_s
                delta_s[n, run] = self.get_delta_output()
                effectivity_s[n, run] = delta_s[n, run]/error_s[n, run]
        
        # Print some statistics
        print ""
        print "N \t gmean(err_u) \t\t gmean(delta_u) \t min(eff_u) \t gmean(eff_u) \t max(eff_u)"
        for n in range(N): # n = 0, 1, ... N - 1
            mean_error_u = np.exp(np.mean(np.log(error_u[n, :])))
            mean_delta_u = np.exp(np.mean(np.log(delta_u[n, :])))
            min_effectivity_u = np.min(effectivity_u[n, :])
            mean_effectivity_u = np.exp(np.mean(np.log(effectivity_u[n, :])))
            max_effectivity_u = np.max(effectivity_u[n, :])
            print str(n+1) + " \t " + str(mean_error_u) + " \t " + str(mean_delta_u) \
                  + " \t " + str(min_effectivity_u) + " \t " + str(mean_effectivity_u) \
                  + " \t " + str(max_effectivity_u)
                  
        print ""
        print "N \t gmean(err_s) \t\t gmean(delta_s) \t min(eff_s) \t gmean(eff_s) \t max(eff_s)"
        for n in range(N): # n = 0, 1, ... N - 1
            mean_error_s = np.exp(np.mean(np.log(error_s[n, :])))
            mean_delta_s = np.exp(np.mean(np.log(delta_s[n, :])))
            min_effectivity_s = np.min(effectivity_s[n, :])
            mean_effectivity_s = np.exp(np.mean(np.log(effectivity_s[n, :])))
            max_effectivity_s = np.max(effectivity_s[n, :])
            print str(n+1) + " \t " + str(mean_error_s) + " \t " + str(mean_delta_s) \
                  + " \t " + str(min_effectivity_s) + " \t " + str(mean_effectivity_s) \
                  + " \t " + str(max_effectivity_s)
        
        print ""
        print "=============================================================="
        print "=             Error analysis ends                            ="
        print "=============================================================="
        print ""
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    def load_reduced_matrices(self):
        # Read in data structures as in parent
        EllipticCoerciveBase.load_reduced_matrices(self)
        # Moreover, read also data structures related to the dual
        if len(np.asarray(self.CC)) == 0: # avoid loading multiple times
            self.CC = np.load(self.dual_folder + "CC.npy")
        if len(np.asarray(self.CL)) == 0: # avoid loading multiple times
            self.CL = np.load(self.dual_folder + "CL.npy")
        if len(np.asarray(self.LL)) == 0: # avoid loading multiple times
            self.LL = np.load(self.dual_folder + "LL.npy")
    
    #  @}
    ########################### end - I/O - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return a lower bound for the coercivity constant
    # example of implementation:
    #    return 1.0
    def get_alpha_lb(self):
        print "The function get_alpha_lb(self) is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function get_alpha_lb(self)!")
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
