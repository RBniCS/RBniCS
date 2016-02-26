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
## @file elliptic_coercive_rb_base.py
#  @brief Implementation of the reduced basis method for (compliant) elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
import os # for path and makedir
import shutil # for rm
import random # to randomize selection in case of equal error bound
from RBniCS.gram_schmidt import GramSchmidt
from RBniCS.elliptic_coercive_base import EllipticCoerciveBase

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE RB BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveRBBase
#
# Base class containing the interface of the RB method
# for (compliant) elliptic coercive problems
class EllipticCoerciveRBBase(EllipticCoerciveBase):
    """This class implements the Certified Reduced Basis Method for
    elliptic and coercive problems. The output of interest are assumed to
    be compliant.

    During the offline stage, the parameters are chosen relying on a
    greedy algorithm. The user must specify how the alpha_lb (i.e., alpha
    lower bound) is computed since this term is needed in the a posteriori
    error estimation. RBniCS features an implementation of the Successive
    Constraints Method (SCM) for the estimation of the alpha_lb (take a
    look at tutorial 4 for the usage of SCM).
    
    The following functions are implemented:

    ## Methods related to the offline stage
    - offline()
    - update_basis_matrix()
    - greedy()
    - compute_dual_terms()
    - compute_a_dual()
    - compute_f_dual()

    ## Methods related to the online stage
    - online_output()
    - get_delta()
    - get_delta_output()
    - get_eps2 ()
    - truth_output()

    ## Error analysis
    - compute_error()
    - error_analysis()
    
    ## Input/output methods
    - load_reduced_matrices()
    
    ## Problem specific methods
    - get_alpha_lb() # to be overridden

    A typical usage of this class is given in the tutorial 1.

    """

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
        self.riesz_A = AffineExpansionOnlineStorage()
        self.riesz_F = AffineExpansionOnlineStorage()
        self.riesz_AA_product = AffineExpansionOnlineStorage()
        self.riesz_AF_product = AffineExpansionOnlineStorage()
        self.riesz_FF_product = AffineExpansionOnlineStorage()
        self.build_error_estimation_matrices.__func__.initialized = False
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 4. Offline output
        self.s = 0
        # 6bis. Declare a GS object
        self.GS = GramSchmidt(self.compute_scalar_product, self.S)
        # 9. I/O
        self.snapshots_folder = "snapshots/"
        self.basis_folder = "basis/"
        self.error_estimation_folder = "error_estimation/"
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
        self.theta_f = self.compute_theta("f")
        assembled_reduced_F = sum(product(self.theta_f, self.reduced_F[:N]))
        self.sN = transpose(assembled_reduced_F)*self.uN
        
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
    def get_eps2(self):
        eps2 = 0.0
        
        # Add the (F, F) product part
        for qf in range(self.Qf):
            for qfp in range(self.Qf):
                eps2 += self.theta_f[qf]*self.theta_f[qfp]*self.riesz_FF_product[qf, qfp]
        
        # Add the (A, F) product part
        for qa in range(self.Qa):
            for qf in range(self.Qf):
                eps2 += 2.0*self.theta_a[qa]*self.theta_f[qf]*self.riesz_AF_product[qa, qf]*self.uN

        # Add the (A, A) product part
        for qa in range(self.Qa):
            for qap in range(self.Qa):
                eps2 += self.theta_a[qa]*self.theta_a[qap]*transpose(self.uN)*self.riesz_AA_product[qa, qap]*self.uN
        
        return eps2
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Initialize data structures required for the offline phase
    def _init_offline(self):
        super(EllipticCoerciveRBBase, self)._init_error_analysis()
        # Also initialize data structures related to error estimation
        self.riesz_A = AffineExpansionOnlineStorage(self.Qa)
        for qa in range(self.Qa):
            self.riesz_A[qa] = FunctionsList()
        self.riesz_F = AffineExpansionOnlineStorage(self.Qf)
        for qf in range(self.Qf):
            self.riesz_F[qf] = FunctionsList() # even though it will be composed of only one function
        self.riesz_AA_product = AffineExpansionOnlineStorage(self.Qa, self.Qa)
        self.riesz_AF_product = AffineExpansionOnlineStorage(self.Qa, self.Qf)
        self.riesz_FF_product = AffineExpansionOnlineStorage(self.Qf, self.Qf)
        
    ## Perform the offline phase of the reduced order model
    def offline(self):
        print("==============================================================")
        print("=             Offline phase begins                           =")
        print("==============================================================")
        print("")
        if os.path.exists(self.post_processing_folder):
            shutil.rmtree(self.post_processing_folder)
        folders = (self.snapshots_folder, self.basis_folder, self.error_estimation_folder, self.reduced_matrices_folder, self.post_processing_folder)
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)
        
        for run in range(self.Nmax):
            print("############################## run = ", run, " ######################################")
            
            print("truth solve for mu = ", self.mu)
            self.truth_solve()
            self.export_solution(self.snapshot, self.snapshots_folder + "truth_" + str(run))
            
            print("update basis matrix")
            self.update_basis_matrix()
            
            print("build reduced matrices")
            self.build_reduced_matrices()
            self.build_reduced_vectors()
            
            print("reduced order solve")
            self._online_solve(self.N)
            
            print("build matrices for error estimation (it may take a while)")
            self.build_error_estimation_matrices()
            
            if self.N < self.Nmax:
                print("find next mu")
            
            # we do a greedy even if N==Nmax in order to have in
            # output the delta_max
            self.greedy()

            print("")
            
        print("==============================================================")
        print("=             Offline phase ends                             =")
        print("==============================================================")
        print("")
        
    ## Update basis matrix
    def update_basis_matrix(self):
        self.Z.enrich(self.snapshot)
        self.GS.apply(self.Z)
        self.Z.save(self.basis_folder, "basis")
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
        print("absolute delta max = ", delta_max)
        self.setmu(munew)
        save_greedy_post_processing_file(self.N, delta_max, munew, self.post_processing_folder)
        
    ## Build matrices for error estimation
    def build_error_estimation_matrices(self):
        if not self.build_error_estimation_matrices.__func__.initialized: # this part does not depend on N, so we compute it only once
            # Compute the Riesz representation of F
            self.compute_riesz_F()
            
            # Compute the (F, F) Riesz representors product
            for qf in range(0, self.Qf):
                for qfp in range(qf, self.Qf):
                    self.riesz_FF_product[qf, qfp] = transpose(self.riesz_F[qf])*self.S*self.riesz_F[qfp]
                    if qf != qfp:
                        self.riesz_FF_product[qfp, qf] = self.riesz_FF_product[qf, qfp]
            self.riesz_FF_product.save(self.error_estimation_folder, "riesz_FF_product")
            
            self.build_error_estimation_matrices.__func__.initialized = True
            
        # Update the Riesz representation of -A*Z with the new basis function(s)
        self.update_riesz_A()
        
        # Update the (A, F) Riesz representors product with the new basis function
        for qa in range(0, self.Qa):
            for qf in range(0, self.Qf):
                self.riesz_AF_product[qa, qf] = transpose(self.riesz_A[qa])*self.S*self.riesz_F[qf]
        self.riesz_AF_product.save(self.error_estimation_folder, "riesz_AF_product")
                
        # Update the (A, A) Riesz representors product with the new basis function
        for qa in range(0, self.Qa):
            for qap in range(qa, self.Qa):
                self.riesz_AA_product[qa, qap] = transpose(self.riesz_A[qa])*self.S*self.riesz_A[qap]
                if qa != qap:
                    self.riesz_AA_product[qap, qa] = self.riesz_AA_product[qa, qap]
        self.riesz_AA_product.save(self.error_estimation_folder, "riesz_AA_product")
    
    ## Compute the Riesz representation of a
    def update_riesz_A(self):
        riesz = Function(self.V)
        for qa in range(self.Qa):
            for n in range(len(self.riesz_A[qa]), self.N):
                solve(self.S, riesz.vector(), -1.*self.truth_A[qa]*self.Z[n])
                self.riesz_A[qa].enrich(riesz.vector())
    
    ## Compute the Riesz representation of f
    def compute_riesz_F(self):
        riesz = Function(self.V)
        for qf in range(self.Qf):
            solve(self.S, riesz.vector(), self.truth_F[qf])
            self.riesz_F[qf].enrich(riesz.vector())
        
    ## Perform a truth evaluation of the output
    def truth_output(self):
        self.theta_f = self.compute_theta("f")
        assembled_truth_F = sum(product(self.theta_f, self.truth_F))
        self.s = transpose(assembled_truth_F)*self.snapshot.vector()
        
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
        if N is None:
            N = self.N
            
        print("==============================================================")
        print("=             Error analysis begins                          =")
        print("==============================================================")
        print("")
        
        error_u = np.zeros((N, len(self.xi_test)))
        delta_u = np.zeros((N, len(self.xi_test)))
        effectivity_u = np.zeros((N, len(self.xi_test)))
        error_s = np.zeros((N, len(self.xi_test)))
        delta_s = np.zeros((N, len(self.xi_test)))
        effectivity_s = np.zeros((N, len(self.xi_test)))
        
        for run in range(len(self.xi_test)):
            print("############################## run = ", run, " ######################################")
            
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
        print("")
        print("N \t gmean(err_u) \t\t gmean(delta_u) \t min(eff_u) \t gmean(eff_u) \t max(eff_u)")
        for n in range(N): # n = 0, 1, ... N - 1
            mean_error_u = np.exp(np.mean(np.log(error_u[n, :])))
            mean_delta_u = np.exp(np.mean(np.log(delta_u[n, :])))
            min_effectivity_u = np.min(effectivity_u[n, :])
            mean_effectivity_u = np.exp(np.mean(np.log(effectivity_u[n, :])))
            max_effectivity_u = np.max(effectivity_u[n, :])
            print(str(n+1) + " \t " + str(mean_error_u) + " \t " + str(mean_delta_u) \
                  + " \t " + str(min_effectivity_u) + " \t " + str(mean_effectivity_u) \
                  + " \t " + str(max_effectivity_u) \
                 )
                  
        print("")
        print("N \t gmean(err_s) \t\t gmean(delta_s) \t min(eff_s) \t gmean(eff_s) \t max(eff_s)")
        for n in range(N): # n = 0, 1, ... N - 1
            mean_error_s = np.exp(np.mean(np.log(error_s[n, :])))
            mean_delta_s = np.exp(np.mean(np.log(delta_s[n, :])))
            min_effectivity_s = np.min(effectivity_s[n, :])
            mean_effectivity_s = np.exp(np.mean(np.log(effectivity_s[n, :])))
            max_effectivity_s = np.max(effectivity_s[n, :])
            print(str(n+1) + " \t " + str(mean_error_s) + " \t " + str(mean_delta_s) \
                  + " \t " + str(min_effectivity_s) + " \t " + str(mean_effectivity_s) \
                  + " \t " + str(max_effectivity_s) \
                 )
        
        print("")
        print("==============================================================")
        print("=             Error analysis ends                            =")
        print("==============================================================")
        print("")
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    def load_reduced_matrices(self):
        # Read in data structures as in parent
        EllipticCoerciveBase.load_reduced_matrices(self)
        # Moreover, read also data structures related to error estimation
        self.CC.load(self.error_estimation_folder, "CC")
        self.CL.load(self.error_estimation_folder, "CL")
        self.LL.load(self.error_estimation_folder, "LL")
            
    ## Save greedy post processing to file
    @staticmethod
    def save_greedy_post_processing_file(N, delta_max, mu_greedy, directory):
        with open(directory + "/delta_max.txt", "a") as outfile:
            file.write(str(N) + " " + str(delta_max))
        with open(directory + "/mu_greedy.txt", "a") as outfile:
            file.write(str(mu_greedy))
        
    #  @}
    ########################### end - I/O - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return a lower bound for the coercivity constant
    # example of implementation:
    #    return 1.0
    def get_alpha_lb(self):
        raise RuntimeError("The function get_alpha_lb(self) is problem-specific and needs to be overridden.")
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
