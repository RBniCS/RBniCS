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
## @file elliptic_coercive_rb_non_compliant_base.py
#  @brief Implementation of the reduced basis method for non compliant elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import numpy as np
import sys # for exit
from elliptic_coercive_rb_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE RB NON COMPLIANT BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveRBNonCompliantBase
#
# Base class containing the interface of the RB method
# for non compliant elliptic coercive problems
class EllipticCoerciveRBNonCompliantBase(EllipticCoerciveRBBase):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, bc_list):
        # Call the parent initialization
        EllipticCoerciveRBBase.__init__(self, V, bc_list)
        
        # Attach a dual problem
        self.dual_problem = _EllipticCoerciveRBNonCompliantBase_Dual(self)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # 3a. Number of terms in the affine expansion
        self.Qs = 0
        # 3b. Theta multiplicative factors of the affine expansion
        self.theta_s = ()
        # 3c. Reduced order matrices/vectors
        self.reduced_S = ()
        self.reduced_A_dp = () # precoumpted expansion of a_q(\phi_j, \psi_i) for \phi_j primal basis function and \psi_i dual basis function
        self.reduced_F_d = () # precoumpted expansion of f_q(\psi_i) for \psi_i dual basis function
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 3c. Matrices/vectors resulting from the truth discretization
        self.truth_S = ()
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    # Propagate the values of all setters also to the dual problem
    
    def setNmax(self, nmax):
        EllipticCoerciveRBBase.setNmax(self, nmax)
        self.dual_problem.setNmax(nmax)
    def settol(self, tol):
        EllipticCoerciveRBBase.settol(self, tol)
        self.dual_problem.settol(tol)
    def setmu_range(self, mu_range):
        EllipticCoerciveRBBase.setmu_range(self, mu_range)
        self.dual_problem.setmu_range(mu_range)
    def setxi_train(self, ntrain, enable_import=False, sampling="random"):
        EllipticCoerciveRBBase.setxi_train(self, ntrain, enable_import, sampling)
        self.dual_problem.setxi_train(ntrain, enable_import, sampling)
    def setxi_test(self, ntest, enable_import=False, sampling="random"):
        EllipticCoerciveRBBase.setxi_test(self, ntest, enable_import, sampling)
        self.dual_problem.setxi_test(ntest, enable_import, sampling)
    def setmu(self, mu):
        EllipticCoerciveRBBase.setmu(self, mu)
        self.dual_problem.setmu(mu)
        
    #  @}
    ########################### end - SETTERS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Perform an online solve. Overridden to solve also the dual problem for output correction
    # and error estimation
    def online_solve(self, N=None, with_plot=True):
        self.dual_problem.online_solve(N, False)
        EllipticCoerciveRBBase.online_solve(self, N, with_plot)
    
    # Perform an online evaluation of the non-compliant output
    def online_output(self):
        N = self.uN.size
        self.sN = 0.
        # Assemble output
        self.theta_s = self.compute_theta_s()
        assembled_reduced_S = self.affine_assemble_reduced_vector(self.reduced_S, self.theta_s, N)
        self.sN += float(np.dot(assembled_reduced_S, self.uN))
        # Assemble correction
        assembled_reduced_A_dp = self.affine_assemble_reduced_matrix(self.reduced_A_dp, self.theta_a, N, N)
        assembled_reduced_F_d = self.affine_assemble_reduced_vector(self.reduced_F_d, self.theta_f, N)
        self.sN -= float(np.dot(assembled_reduced_F_d, self.dual_problem.uN)) - float(np.matrix(self.dual_problem.uN.T)*(assembled_reduced_A_dp*np.matrix(self.uN)))
    
    ## Return an error bound for the current solution. Overridden to be computed in the V-norm
    #  since the energy norm is not defined generally in the non compliant case
    def get_delta(self):
        eps2 = self.get_eps2()
        alpha = self.get_alpha_lb()
        return np.sqrt(np.abs(eps2))/alpha
    
    ## Return an error bound for the current output
    def get_delta_output(self):
        primal_eps2 = self.get_eps2()
        dual_eps2 = self.dual_problem.get_eps2()
        alpha = self.get_alpha_lb()
        return np.sqrt(np.abs(primal_eps2*dual_eps2))/alpha
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
    def offline(self):
        self.truth_S = self.assemble_truth_s()
        self.apply_bc_to_vector_expansion(self.truth_S)
        self.Qs = len(self.truth_S)
        
        # Perform the offline stage of the dual problem
        bak_first_mu = tuple(list(self.mu))
        self.dual_problem.offline()
        
        # Perform the offline stage of the primal problem
        self.setmu(bak_first_mu)
        EllipticCoerciveRBBase.offline(self)
        
    ## Perform a truth evaluation of the output
    def truth_output(self):
        self.theta_s = self.compute_theta_s()
        assembled_truth_S = self.affine_assemble_truth_vector(self.truth_S, self.theta_s)
        self.s = assembled_truth_S.inner(self.snapshot.vector())
    
    ## Assemble the reduced order affine expansion (matrix). Overridden to assemble also terms related to output correction
    def build_reduced_matrices(self):
        EllipticCoerciveRBBase.build_reduced_matrices(self)
        
        # Output correction terms
        reduced_A_dp = ()
        for A in self.truth_A:
            A = as_backend_type(A)
            dim = A.size(0) # = A.size(1)
            if self.N == 1:
                reduced_A_dp += (np.dot(self.dual_problem.Z.T,A.mat().getValues(range(dim),range(dim)).dot(self.Z)),)
            else:
                reduced_A_dp += (np.matrix(np.dot(self.dual_problem.Z.T,np.matrix(np.dot(A.mat().getValues(range(dim),range(dim)),self.Z)))),)
        self.reduced_A_dp = reduced_A_dp
        np.save(self.reduced_matrices_folder + "reduced_A_dp", self.reduced_A_dp)
    
    ## Assemble the reduced order affine expansion (rhs). Overridden to assemble also terms related to output  and output correction
    def build_reduced_vectors(self):
        EllipticCoerciveRBBase.build_reduced_vectors(self)
        
        # Output terms
        reduced_S = ()
        for S in self.truth_S:
            S = as_backend_type(S)
            dim = S.size()
            reduced_s = np.dot(self.Z.T, S.vec().getValues(range(dim)))
            reduced_S += (reduced_s,)
        self.reduced_S = reduced_S
        np.save(self.reduced_matrices_folder + "reduced_S", self.reduced_S)
        
        # Output correction terms
        reduced_F_d = ()
        for F in self.truth_F:
            F = as_backend_type(F)
            dim = F.size()
            reduced_f_d = np.dot(self.dual_problem.Z.T, F.vec().getValues(range(dim)))
            reduced_F_d += (reduced_f_d,)
        self.reduced_F_d = reduced_F_d
        np.save(self.reduced_matrices_folder + "reduced_F_d", self.reduced_F_d)
        
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu. Overridden to compute the error in the V-norm
    def compute_error(self, N=None, skip_truth_solve=False):
        if not skip_truth_solve:
            self.truth_solve()
            self.truth_output()
        self.online_solve(N, False)
        self.online_output()
        self.error.vector()[:] = self.snapshot.vector()[:] - self.reduced.vector()[:] # error as a function
        error_u_norm_squared = self.compute_scalar(self.error, self.error, self.S) # norm of the error
        error_u_norm = np.sqrt(error_u_norm_squared)
        error_s = abs(self.s - self.sN)
        return (error_u_norm, error_s)
        
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    def error_analysis(self, N=None):
        self.truth_S = self.assemble_truth_s()
        self.apply_bc_to_vector_expansion(self.truth_S)
        self.Qs = len(self.truth_S)
        
        # Perform the error analysis of the dual problem
        self.dual_problem.error_analysis(N)
        # Perform the error analysis of the primal problem
        EllipticCoerciveRBBase.error_analysis(self, N)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    def load_reduced_matrices(self):
        # Read in data structures as in parent
        EllipticCoerciveRBBase.load_reduced_matrices(self)
        # Moreover, read also data structures related to the dual problem
        self.dual_problem.load_reduced_matrices()
        # ... and those related to output and output correction
        if len(np.asarray(self.reduced_A_dp)) == 0: # avoid loading multiple times
            self.reduced_A_dp = tuple(np.load(self.reduced_matrices_folder + "reduced_A_dp.npy"))
        if len(np.asarray(self.reduced_S)) == 0: # avoid loading multiple times
            self.reduced_S = tuple(np.load(self.reduced_matrices_folder + "reduced_S.npy"))
        if len(np.asarray(self.reduced_F_d)) == 0: # avoid loading multiple times
            self.reduced_F_d = tuple(np.load(self.reduced_matrices_folder + "reduced_F_d.npy"))
    
    #  @}
    ########################### end - I/O - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return theta multiplicative terms of the affine expansion of s.
    # example of implementation:
    #    m1 = self.mu[0]
    #    m2 = self.mu[1]
    #    m3 = self.mu[2]
    #    theta_s0 = m1
    #    theta_s1 = m2
    #    theta_s2 = m1*m2+m3/7.0
    #    return (theta_s0, theta_s1, theta_s2)
    def compute_theta_s(self):
        print "The function compute_theta_s() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function compute_theta_s(self)!")
        
    ## Return vectors resulting from the truth discretization of s.
    #    s0 = v*ds(1)
    #    S0 = assemble(S0)
    #    return (S0,)
    def assemble_truth_s(self):
        print "The function compute_truth_s() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function assemble_truth_s(self)!")
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE RB NON COMPLIANT: AUXILIARY CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class _EllipticCoerciveRBNonCompliantBase_Dual
#
# Class containing the dual problem
class _EllipticCoerciveRBNonCompliantBase_Dual(EllipticCoerciveRBBase):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    def __init__(self, primal_problem):
        # Call the parent initialization
        EllipticCoerciveRBBase.__init__(self, primal_problem.V, primal_problem.bc_list)
        self.primal_problem = primal_problem
        
        # Possibly copy the inner product matrix, if the primal problem has redefined it
        self.S = self.primal_problem.S
        
        # 9. I/O
        self.xi_train_folder = "xi_train__dual/"
        self.xi_test_folder = "xi_test__dual/"
        self.snapshots_folder = "snapshots__dual/"
        self.basis_folder = "basis__dual/"
        self.dual_folder = "dual__dual/"
        self.reduced_matrices_folder = "reduced_matrices__dual/"
        self.post_processing_folder = "post_processing__dual/"
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Return an error bound for the current solution. Overridden to be computed in the V-norm
    #  since the energy norm is not defined generally in the non compliant case
    def get_delta(self):
        eps2 = self.get_eps2()
        alpha = self.get_alpha_lb()
        return np.sqrt(np.abs(eps2))/alpha
        
    #  @}
    ########################### end - ONLINE STAGE - end ###########################
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu. Overridden to compute the error in the V-norm
    def compute_error(self, N=None, skip_truth_solve=False):
        if not skip_truth_solve:
            self.truth_solve()
        self.online_solve(N, False)
        self.error.vector()[:] = self.snapshot.vector()[:] - self.reduced.vector()[:] # error as a function
        error_norm_squared = self.compute_scalar(self.error, self.error, self.S) # norm of the error
        return np.sqrt(error_norm_squared)
        
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    def error_analysis(self, N=None):
        # Possibly need to initialize Qa of primal, since error_analysis of dual
        # may be performed before any primal data structures is initialized,
        # but we may rely on the primal itself in get_alpha_lb, when querying SCM
        self.primal_problem.theta_a = self.primal_problem.compute_theta_a()
        self.primal_problem.Qa = len(self.primal_problem.theta_a)
        # This is almost the same as in parent, without the output computation,
        # since it makes no sense here.
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
        
        for run in range(len(self.xi_test)):
            print "############################## run = ", run, " ######################################"
            
            self.setmu(self.xi_test[run])
            
            # Perform the truth solve only once
            self.truth_solve()
            self.truth_output()
            
            for n in range(N): # n = 0, 1, ... N - 1
                current_error_u = self.compute_error(n + 1, True)
                
                error_u[n, run] = current_error_u
                delta_u[n, run] = self.get_delta()
                effectivity_u[n, run] = delta_u[n, run]/error_u[n, run]
                
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
        print "=============================================================="
        print "=             Error analysis ends                            ="
        print "=============================================================="
        print ""
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
        
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return the alpha_lower bound.
    def get_alpha_lb(self):
        self.primal_problem.setmu(self.mu)
        return self.primal_problem.get_alpha_lb()
    
    ## Set theta multiplicative terms of the affine expansion of a.
    def compute_theta_a(self):
        self.primal_problem.setmu(self.mu)
        return self.primal_problem.compute_theta_a()
    
    ## Set theta multiplicative terms of the affine expansion of f.
    def compute_theta_f(self):
        self.primal_problem.setmu(self.mu)
        primal_theta_s = self.primal_problem.compute_theta_s()
        primal_theta_s_minus = ()
        for qs in range(len(primal_theta_s)):
            primal_theta_s_minus += (- primal_theta_s[qs],)
        return primal_theta_s_minus
    
    ## Set matrices resulting from the truth discretization of a.
    def assemble_truth_a(self):
        primal_truth_a = self.primal_problem.assemble_truth_a()
        primal_truth_a_transpose = ()
        for qa in range(len(primal_truth_a)):
            primal_truth_a_qa_transpose = self.compute_transpose(primal_truth_a[qa])
            primal_truth_a_transpose += (primal_truth_a_qa_transpose,)
        return primal_truth_a_transpose
    
    ## Set vectors resulting from the truth discretization of f.
    def assemble_truth_f(self):
        return self.primal_problem.assemble_truth_s()
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Deform the mesh as a function of the geometrical parameters
    def move_mesh(self):
        self.primal_problem.setmu(self.mu)
        self.primal_problem.move_mesh()
    
    ## Restore the reference mesh
    def reset_reference(self):
        self.primal_problem.reset_reference()
                
    #  @}
    ########################### end - I/O - end ########################### 
