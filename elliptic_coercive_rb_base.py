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
## @file elliptic_coercive_rb_base.py
#  @brief Implementation of the reduced basis method for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import os # for path and makedir
import shutil # for rm
import sys # for exit
from scipy import stats as scistats # for geometric mean
import random # to randomize selection in case of equal error bound
from elliptic_coercive_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE RB BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveBase
#
# Base class containing the interface of the RB method
# for elliptic coercive problems
class EllipticCoerciveRBBase(EllipticCoerciveBase):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V):
        # Call the parent initialization
        EllipticCoerciveBase.__init__(self, V)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # 4. Residual terms
        self.Cf = ()
        self.CC = ()
        self.CL = ()
        self.LL = ()
        self.lnq = ()
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 9. I/O
        self.snap_folder = "snapshots/"
        self.basis_folder = "basis/"
        self.dual_folder = "dual/"
        self.red_matrices_folder = "red_matr/"
        self.pp_folder = "pp/" # post processing
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Return an error bound for the current solution
    def get_delta(self):
        eps2 = self.get_eps2()
        alpha = self.get_alpha_lb()
        return np.sqrt(np.abs(eps2)/alpha)
    
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
            n=0
            for un in uN:
                for qf in range(Qf):
                    for qa in range(Qa):
                        eps2 += 2.0* theta_f[qf]*theta_a[qa]*un*CL[n][qf,qa]
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
        if os.path.exists(self.pp_folder):
            shutil.rmtree(self.pp_folder)
        folders = (self.snap_folder, self.basis_folder, self.dual_folder, self.red_matrices_folder, self.pp_folder)
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)
        
        self.truth_A = self.assemble_truth_a()
        self.truth_F = self.assemble_truth_f()
        self.Qa = len(self.truth_A)
        self.Qf = len(self.truth_F)
        
        for run in range(self.Nmax):
            print "############################## run = ", run, " ######################################"
            
            print "truth solve for mu = ", self.mu
            self.truth_solve()
            self.export_solution(self.snap, self.snap_folder + "truth_" + str(run))
            
            print "update basis matrix"
            self.update_basis_matrix()
            
            print "build reduced matrices"
            self.build_red_matrices()
            self.build_red_vectors()
            
            print "solve rb"
            self.red_solve(self.N)
            
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
            self.Z = np.array(self.snap.vector()).reshape(-1, 1) # as column vector
            self.Z /= np.sqrt(np.dot(self.Z[:, 0], self.S*self.Z[:, 0]))
    
        else:
            self.Z = np.hstack((self.Z, self.snap.vector())) # add new basis functions as column vectors
            self.Z = self.GS()
        np.save(self.basis_folder + "basis", self.Z)
        current_basis = Function(self.V)
        current_basis.vector()[:] = self.Z[:, self.N]
        self.export_basis(current_basis, self.basis_folder + "basis_" + str(self.N))
        self.N += 1
    
    ## Perform Gram Schmidt orthonormalization
    def GS(self):
        basis = self.Z
        last = basis.shape[1]-1
        b = basis[:, last].copy()
        for i in range(last):
            proj = np.dot(np.dot(b,self.S*basis[:, i])/np.dot(basis[:, i],self.S*basis[:, i]),basis[:, i])
            b = b - proj 
        basis[:, last] = b/np.sqrt(np.dot(b,self.S*b))
        return basis
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        delta_max = -1.0
        for mu in self.xi_train:
            self.setmu(mu)
            self.red_solve(self.N)
            delta = self.get_delta()
            if (delta > delta_max or (delta == delta_max and random.random() >= 0.5)):
                delta_max = delta
                munew = mu
        print "absolute delta max = ", delta_max
        if os.path.isfile(self.pp_folder + "delta_max.npy") == True:
            d = np.load(self.pp_folder + "delta_max.npy")
            
            np.save(self.pp_folder + "delta_max", np.append(d, delta_max))
    
            m = np.load(self.pp_folder + "mu_greedy.npy")
            np.save(self.pp_folder + "mu_greedy", np.append(m, munew))
        else:
            np.save(self.pp_folder + "delta_max", delta_max)
            np.save(self.pp_folder + "mu_greedy", np.array(munew))

        self.setmu(munew)
        
    ## Compute dual terms
    def compute_dual_terms(self):
        N = self.N
        RBu = Function(self.V)
        
        Qf = self.Qf
        Qa = self.Qa
        if self.N == 1 :
            
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
            self.CL = np.zeros((self.Nmax,self.Qf,self.Qa))
            for qf in range(0,Qf):
                for qa in range(0,Qa):
                    la.vector()[:] = self.lnq[0][:, qa]
                    self.CL[0,qf,qa] = self.compute_scalar(la,self.Cf[qf],self.S)
            np.save(self.dual_folder + "CL", self.CL)
            
            # LL
            self.LL = np.zeros((self.Nmax,self.Nmax,self.Qa,self.Qa))
            for qa in range(0,Qa):
                la.vector()[:] = self.lnq[0][:, qa]
                for qap in range(qa,Qa):
                    lap.vector()[:] = self.lnq[0][:, qap]
                    self.LL[0,0,qa,qap] = self.compute_scalar(la,lap,self.S)
                    if qa != qap:
                        self.LL[0,0,qap,qa] = self.LL[0,0,qa,qap]
            np.save(self.dual_folder + "LL", self.LL)
        else:
            RBu.vector()[:] = self.Z[:, N-1]
            self.lnq += (self.compute_a_dual(RBu),)
            la = Function(self.V)
            lap = Function(self.V)
            cl = np.zeros((Qf,Qa))
            n = self.N-1

            # CL
            for qf in range(0,Qf):
                for qa in range(0,Qa):
                    la.vector()[:] = self.lnq[N-1][:, qa]
                    self.CL[n,qf,qa] = self.compute_scalar(self.Cf[qf],la,self.S)
            np.save(self.dual_folder + "CL", self.CL)
    
            # LL
            for qa in range(0,Qa):
                la.vector()[:] = self.lnq[n][:, qa]
                for nn in range(0,N):
                    for qap in range(0,Qa):
                        lap.vector()[:] = self.lnq[nn][:, qap]
                        self.LL[n,nn,qa,qap] = self.compute_scalar(la,lap,self.S)
                        if n != nn:
                            self.LL[nn,n,qa,qap] = self.LL[n,nn,qa,qap]
            np.save(self.dual_folder + "LL", self.LL)
    
    ## Compute the dual of a
    def compute_a_dual(self, RBu):
        riez = Function(self.V)
        i = 0
        for A in self.truth_A:
            solve (self.S,riez.vector(), A*RBu.vector()*(-1.0))
            if i != 0:
                l = np.hstack((l,riez.vector()))
            else:
                l = np.array(riez.vector()).reshape(-1, 1) # as column vector
                i = 1
        return l
    
    ## Compute the dual of f
    def compute_f_dual(self):
        riez = Function(self.V)
        c = ()
        for F in self.truth_F:
            solve (self.S, riez.vector(), F)
            c += (riez.copy(True),)
        return c
        
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
        
        # Make sure to regenerate a new test set (necessarily random), 
        # otherwise effectivity -> +\infty at greedily selected points
        self.setxi_train(len(self.xi_train))
        
        error = np.zeros((N, len(self.xi_train)))
        delta = np.zeros((N, len(self.xi_train)))
        effectivity = np.zeros((N, len(self.xi_train)))
        
        for run in range(len(self.xi_train)):
            print "############################## run = ", run, " ######################################"
            
            self.setmu(self.xi_train[run])
            
            # Perform the truth solve only once
            self.truth_solve()
            
            for n in range(N): # n = 0, 1, ... N - 1
                error[n, run] = self.compute_error(n + 1, False)
                delta[n, run] = self.get_delta()
                effectivity[n, run] = delta[n, run]/error[n, run]
        
        # Print some statistics
        print ""
        print "N \t gmean(err) \t\t gmean(delta) \t\t gmean(eff) \t max(eff)"
        for n in range(N): # n = 0, 1, ... N - 1
            mean_error = scistats.gmean(error[n, :])
            mean_delta = scistats.gmean(delta[n, :])
            mean_effectivity = scistats.gmean(effectivity[n, :])
            max_effectivity = np.max(effectivity[n, :])
            print str(n+1) + " \t " + str(mean_error) + " \t " + str(mean_delta) \
                  + " \t " + str(mean_effectivity) + " \t " + str(max_effectivity)
        
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
    
    def load_red_matrices(self):
        # Read in data structures as in parent
        EllipticCoerciveBase.load_red_matrices(self)
        # Moreover, read also data structures related to the dual
        if not self.CC.size: # avoid loading multiple times
            self.CC = np.load(self.dual_folder + "CC.npy")
        if not self.CL.size: # avoid loading multiple times
            self.CL = np.load(self.dual_folder + "CL.npy")
        if not self.LL.size: # avoid loading multiple times
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
