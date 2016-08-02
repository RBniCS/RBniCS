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
## @file elliptic_coercive_rb_non_compliant.py
#  @brief Implementation of the reduced basis method for non compliant elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.utils.mpi import print

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE RB NON COMPLIANT BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveRBNonCompliant
#
# Base class containing the interface of the RB method
# for non compliant elliptic coercive problems
class EllipticCoerciveRBNonCompliant(EllipticCoerciveRB):
    """This class relaxes the hypotesis of compliant output for elliptic
    coercive problems. Basically, for each instance of the parameter,
    TWO problems are solved: the problem itself and its adjoint
    one. The two problems are cooked together in order to have a
    better a posteriori error estimation to be used for the
    certification of the ouput as well as for the parameter space
    exploration.

    A typical usage of this class is provided in tutorial 4.

    """

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    def __init__(self, truth_problem):
        # Call the parent initialization
        EllipticCoerciveRB.__init__(self, truth_problem)
        
        # Attach a dual problem
        self.dual_problem = _EllipticCoerciveRBNonCompliant_Dual(self)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Number of terms in the affine expansion
        self.Qs = 0
        # Reduced order operators
        self.operator_s = tuple()
        self.operator_a_dp = tuple() # precoumpted expansion of a_q(\phi_j, \psi_i) for \phi_j primal basis function and \psi_i dual basis function
        self.operator_f_d = tuple() # precoumpted expansion of f_q(\psi_i) for \psi_i dual basis function
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Matrices/vectors resulting from the truth discretization
        self.truth_S = tuple()
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    # Propagate the values of all setters also to the dual problem
    
    def set_Nmax(self, Nmax, **kwargs):
        EllipticCoerciveRB.set_Nmax(self, Nmax, **kwargs)
        self.dual_problem.set_Nmax(Nmax, **kwargs) # TODO are kwargs needed?
    def set_mu_range(self, mu_range):
        EllipticCoerciveRB.set_mu_range(self, mu_range)
        self.dual_problem.set_mu_range(mu_range)
    def set_xi_train(self, ntrain, enable_import=True, sampling=None):
        EllipticCoerciveRB.set_xi_train(self, ntrain, enable_import, sampling)
        self.dual_problem.set_xi_train(ntrain, enable_import, sampling)
    def set_xi_test(self, ntest, enable_import=False, sampling=None):
        EllipticCoerciveRB.set_xi_test(self, ntest, enable_import, sampling)
        self.dual_problem.set_xi_test(ntest, enable_import, sampling)
    def set_mu(self, mu):
        EllipticCoerciveRB.set_mu(self, mu)
        self.dual_problem.set_mu(mu)
        
    #  @}
    ########################### end - SETTERS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Perform an online solve. Overridden to solve also the dual problem for output correction
    # and error estimation
    def online_solve(self, N=None, with_plot=True):
        self.dual_problem.online_solve(N, False)
        EllipticCoerciveRB.online_solve(self, N, with_plot)
    
    # Perform an online evaluation of the non-compliant output
    def online_output(self):
        N = self.uN.size
        self.sN = 0.
        # Assemble output
        assembled_operator_s = sum(product(self.compute_theta("s"), self.operator_s[:N]))
        self.sN += transpose(assembled_operator_s)*self.uN
        # Assemble correction
        assembled_operator_a_dp = sum(product(self.compute_theta("a"), self.operator_a_dp[:N, :N]))
        assembled_operator_f_d = sum(product(self.compute_theta("f"), self.operator_f_d[:N]))
        self.sN -= transpose(assembled_operator_f_d)*self.dual_problem.uN - transpose(self.dual_problem.uN)*assembled_operator_a_dp*self.uN
    
    ## Return an error bound for the current solution. Overridden to be computed in the V-norm
    #  since the energy norm is not defined generally in the non compliant case
    def estimate_error(self):
        eps2 = self.get_residual_norm_squared()
        alpha = self.get_stability_factor()
        return np.sqrt(np.abs(eps2))/alpha
    
    ## Return an error bound for the current output
    def estimate_error_output(self):
        primal_eps2 = self.get_residual_norm_squared()
        dual_eps2 = self.dual_problem.get_residual_norm_squared()
        alpha = self.get_stability_factor()
        return np.sqrt(np.abs(primal_eps2*dual_eps2))/alpha
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
    def offline(self):
        self.truth_S = [assemble(s_form) for s_form in self.assemble_truth_s()]
        self.apply_bc_to_vector_expansion(self.truth_S)
        self.Qs = len(self.truth_S)
        
        # Perform the offline stage of the dual problem
        bak_first_mu = tuple(list(self.mu))
        self.dual_problem.offline()
        
        # Perform the offline stage of the primal problem
        self.set_mu(bak_first_mu)
        EllipticCoerciveRB.offline(self)
        
    ## Perform a truth evaluation of the output
    def truth_output(self):
        assembled_truth_S = sum(product(self.compute_theta("s"), self.truth_S))
        self.s = transpose(assembled_truth_S)*self.snapshot.vector()
    
    ## Assemble the reduced order affine expansion. Overridden to assemble also terms related to output and output correction
    def build_reduced_operators(self):
        EllipticCoerciveRB.build_reduced_operators(self)
        
        # Output correction terms
        operator_a_dp = AffineExpansionOnlineStorage(self.Qa)
        for qa in range(self.Qa):
            operator_a_dp[qa] = transpose(self.dual_problem.Z)*self.truth_A[qa]*self.Z
        self.operator_a_dp = operator_a_dp
        np.save(self.reduced_operators_folder + "operator_a_dp", self.operator_a_dp)
        
        operator_f_d = AffineExpansionOnlineStorage(self.Qf)
        for qf in range(self.Qf):
            operator_f_d[qf] = transpose(self.dual_problem.Z)*self.truth_F[qf]
        self.operator_f_d = operator_f_d
        np.save(self.reduced_operators_folder + "operator_f_d", self.operator_f_d)
        
        # Output terms
        operator_s = AffineExpansionOnlineStorage(self.Qs)
        for qs in range(self.Qs):
            operator_s[qs] = transpose(self.Z)*self.truth_S[qs]
        self.operator_s = operator_s
        np.save(self.reduced_operators_folder + "operator_s", self.operator_s)
                
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu. Overridden to compute the error in the V-norm
    def compute_error(self, N=None, skip_truth_solve=False):
    # TODO update
        if not skip_truth_solve:
            self.truth_solve()
            self.truth_output()
        self.online_solve(N, False)
        self.online_output()
        self.error.vector()[:] = self.snapshot.vector()[:] - self.reduced.vector()[:] # error as a function
        error_u_norm_squared = transpose(self.error)*self.S*self.error # norm of the error
        error_u_norm = np.sqrt(error_u_norm_squared)
        error_s = abs(self.s - self.sN)
        return (error_u_norm, error_s)
        
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    def error_analysis(self, N=None):
        self.truth_S = [assemble(s_form) for s_form in self.assemble_truth_s()]
        self.apply_bc_to_vector_expansion(self.truth_S)
        self.Qs = len(self.truth_S)
        
        # Perform the error analysis of the dual problem
        self.dual_problem.error_analysis(N)
        # Perform the error analysis of the primal problem
        EllipticCoerciveRB.error_analysis(self, N)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    def load_reduced_data_structures(self):
        # Read in data structures as in parent
        EllipticCoerciveRB.load_reduced_data_structures(self)
        # Moreover, read also data structures related to the dual problem
        self.dual_problem.load_reduced_data_structures()
        # ... and those related to output and output correction
        if len(np.asarray(self.operator_a_dp)) == 0: # avoid loading multiple times
            self.operator_a_dp = tuple(np.load(self.reduced_operators_folder + "operator_a_dp.npy"))
        if len(np.asarray(self.operator_s)) == 0: # avoid loading multiple times
            self.operator_s = tuple(np.load(self.reduced_operators_folder + "operator_s.npy"))
        if len(np.asarray(self.operator_f_d)) == 0: # avoid loading multiple times
            self.operator_f_d = tuple(np.load(self.reduced_operators_folder + "operator_f_d.npy"))
    
    #  @}
    ########################### end - I/O - end ########################### 

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE RB NON COMPLIANT: AUXILIARY CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class _EllipticCoerciveRBNonCompliant_Dual
#
# Class containing the dual problem
class _EllipticCoerciveRBNonCompliant_Dual(EllipticCoerciveRB):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    def __init__(self, primal_problem):
        # Call the parent initialization
        EllipticCoerciveRB.__init__(self, primal_problem)
        self.primal_problem = primal_problem
        
        # Possibly copy the inner product matrix, if the primal problem has redefined it
        self.S = self.primal_problem.S
        
        # I/O
        self.xi_train_folder = "xi_train__dual"
        self.xi_test_folder = "xi_test__dual"
        self.snapshots_folder = "snapshots__dual"
        self.basis_folder = "basis__dual"
        self.error_estimation_folder = "error_estimation__dual"
        self.reduced_operators_folder = "reduced_matrices__dual"
        self.post_processing_folder = "post_processing__dual"
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Return an error bound for the current solution. Overridden to be computed in the V-norm
    #  since the energy norm is not defined generally in the non compliant case
    def estimate_error(self):
        eps2 = self.get_residual_norm_squared()
        alpha = self.get_stability_factor()
        return np.sqrt(np.abs(eps2))/alpha
        
    #  @}
    ########################### end - ONLINE STAGE - end ###########################
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu. Overridden to compute the error in the V-norm
    def compute_error(self, truth_solution_and_output, N=None):
    # TODO update
        if not skip_truth_solve:
            self.truth_solve()
        self.online_solve(N, False)
        self.error.vector()[:] = self.snapshot.vector()[:] - self.reduced.vector()[:] # error as a function
        error_norm_squared = self.compute_scalar_product(self.error, self.S, self.error) # norm of the error
        return np.sqrt(error_norm_squared)
        
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    def error_analysis(self, N=None):
        # Possibly need to initialize Qa of primal, since error_analysis of dual
        # may be performed before any primal data structures is initialized,
        # but we may rely on the primal itself in get_stability_factor, when querying SCM
        self.primal_problem.Qa = len(self.primal_problem.compute_theta("a"))
        # This is almost the same as in parent, without the output computation,
        # since it makes no sense here.
        self.load_reduced_data_structures()
        if N is None:
            N = self.N
            
        self.truth_A = [assemble(a_form) for a_form in self.assemble_truth_a()]
        self.apply_bc_to_matrix_expansion(self.truth_A)
        self.truth_F = [assemble(f_form) for f_form in self.assemble_truth_f()]
        self.apply_bc_to_vector_expansion(self.truth_F)
        self.Qa = len(self.truth_A)
        self.Qf = len(self.truth_F)
        
        print("==============================================================")
        print("=             Error analysis begins                          =")
        print("==============================================================")
        print("")
        
        error_u = np.zeros((N, len(self.xi_test)))
        error_estimator_u = np.zeros((N, len(self.xi_test)))
        effectivity_u = np.zeros((N, len(self.xi_test)))
        
        for run in range(len(self.xi_test)):
            print("############################## run =", run, "######################################")
            
            self.set_mu(self.xi_test[run])
            
            # Perform the truth solve only once
            self.truth_solve()
            self.truth_output()
            
            for n in range(N): # n = 0, 1, ... N - 1
                current_error_u = self.compute_error(n + 1, True)
                
                error_u[n, run] = current_error_u
                error_estimator_u[n, run] = self.estimate_error()
                effectivity_u[n, run] = error_estimator_u[n, run]/error_u[n, run]
                
        # Print some statistics
        print("")
        print("N \t gmean(err_u) \t\t gmean(error_estimator_u) \t min(eff_u) \t gmean(eff_u) \t max(eff_u)")
        for n in range(N): # n = 0, 1, ... N - 1
            mean_error_u = np.exp(np.mean(np.log(error_u[n, :])))
            mean_error_estimator_u = np.exp(np.mean(np.log(error_estimator_u[n, :])))
            min_effectivity_u = np.min(effectivity_u[n, :])
            mean_effectivity_u = np.exp(np.mean(np.log(effectivity_u[n, :])))
            max_effectivity_u = np.max(effectivity_u[n, :])
            print(str(n+1) + " \t " + str(mean_error_u) + " \t " + str(mean_error_estimator_u) \
                  + " \t " + str(min_effectivity_u) + " \t " + str(mean_effectivity_u) \
                  + " \t " + str(max_effectivity_u) \
                 )
                  
        print("")
        print("==============================================================")
        print("=             Error analysis ends                            =")
        print("==============================================================")
        print("")
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
        
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return the alpha_lower bound.
    def get_stability_factor(self):
        self.primal_problem.set_mu(self.mu)
        return self.primal_problem.get_stability_factor()
    
    ## Set theta multiplicative terms of the affine expansion of a.
    # TODO update
    def compute_theta_a(self):
        self.primal_problem.set_mu(self.mu)
        return self.primal_problem.compute_theta("a")
    
    ## Set theta multiplicative terms of the affine expansion of f.
    # TODO update
    def compute_theta_f(self):
        self.primal_problem.set_mu(self.mu)
        primal_theta_s = self.primal_problem.compute_theta("s")
        primal_theta_s_minus = tuple()
        for qs in range(len(primal_theta_s)):
            primal_theta_s_minus += (- primal_theta_s[qs],)
        return primal_theta_s_minus
    
    ## Set matrices resulting from the truth discretization of a.
    def assemble_truth_a(self):
        return [adjoint(a_form) for a_form in self.primal_problem.assemble_truth_a()]
    
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
        self.primal_problem.set_mu(self.mu)
        self.primal_problem.move_mesh()
    
    ## Restore the reference mesh
    def reset_reference(self):
        self.primal_problem.reset_reference()
                
    #  @}
    ########################### end - I/O - end ###########################
