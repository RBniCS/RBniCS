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
## @file elliptic_coercive_rb_non_compliant_base.py
#  @brief Implementation of the reduced basis method for non compliant elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import os # for path and makedir
import shutil # for rm
import sys # for exit
from scipy import stats as scistats # for geometric mean
import random # to randomize selection in case of equal error bound
from gram_schmidt import *
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
        self.red_S = ()
        self.red_A_pd = () # precoumpted expansion of a_q(\phi_j, \psi_i) for \phi_j primal basis function and \psi_i dual basis function
        self.red_F_d = () # precoumpted expansion of f_q(\psi_i) for \psi_i dual basis function
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 3c. Matrices/vectors resulting from the truth discretization
        self.truth_A = ()
        self.truth_F = ()
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Perform an online solve. Overridden to solve also the dual problem for output correction
    # and error estimation
    def online_solve(self, N=None, with_plot=True):
        self.dual_problem.online_solve(N, False)
        EllipticCoerciveRBBase.online_solve(N, with_plot)
    
    # Perform an online evaluation of the non-compliant output
    def online_output(self):
        N = self.uN.size
        assembled_red_S = self.aff_assemble_red_vector(self.red_S, self.theta_s, N)
        self.sN = np.dot(assembled_red_S, self.uN)
    
    ## Return an error bound for the current output
    def get_delta_output(self):
        return self.get_delta()*self.dual_problem.get_delta()
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
    def offline(self):
        self.truth_S = self.assemble_truth_s()
        self.apply_bc_to_matrix_expansion(self.truth_S)
        
        # Perform the offline stage of the dual problem
        self.dual_problem.offline()
        
        # Perform the offline stage of the primal problem
        EllipticCoerciveRBBase.offline(self)
        
    ## Perform a truth evaluation of the output
    def truth_output(self):
        self.theta_s = self.compute_theta_s()
        assembled_truth_S = self.aff_assemble_truth_vector(self.truth_S, self.theta_s)
        self.s = assembled_truth_S.inner(self.snap.vector())
    
    ## Assemble the reduced order affine expansion (matrix). Overridden to assemble also terms related to output correction
    def build_red_matrices(self):
        EllipticCoerciveRBBase.build_red_matrices(self)
        
        # Output correction terms
        red_A_pd = ()
        for A in self.truth_A:
            A = as_backend_type(A)
            dim = A.size(0) # = A.size(1)
            if self.N == 1:
                red_A_pd += (np.dot(self.Z.T,A.mat().getValues(range(dim),range(dim)).dot(self.dual_problem.Z)),)
            else:
                red = np.matrix(np.dot(self.Z.T,np.matrix(np.dot(A.mat().getValues(range(dim),range(dim)),self.dual_problem.Z))))
                red_A_pd += (red,)
        self.red_A_pd = red_A_pd
        np.save(self.red_matrices_folder + "red_A_pd", self.red_A_pd)
    
    ## Assemble the reduced order affine expansion (rhs). Overridden to assemble also terms related to output  and output correction
    def build_red_vectors(self):
        EllipticCoerciveRBBase.build_red_vectors(self)
        
        # Output terms
        red_S = ()
        for S in self.truth_S:
            S = as_backend_type(S)
            dim = S.size()
            red_s = np.dot(self.Z.T, S.vec().getValues(range(dim)) )
            red_S += (red_s,)
        self.red_S = red_S
        np.save(self.red_matrices_folder + "red_S", self.red_S)
        
        # Output correction terms
        red_F_d = ()
        for F in self.truth_F:
            F = as_backend_type(F)
            dim = F.size()
            red_f_d = np.dot(self.dual_problem.Z.T, F.vec().getValues(range(dim)) )
            red_F_d += (red_f_d,)
        self.red_F_d = red_F_d
        np.save(self.red_matrices_folder + "red_F_d", self.red_F_d)
        
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    def load_red_matrices(self):
        # Read in data structures as in parent
        EllipticCoerciveRBBase.load_red_matrices(self)
        # Moreover, read also data structures related to the dual problem
        self.dual_problem.load_red_matrices(self)
        # ... and those related to output and output correction
        if not self.CC.size: # avoid loading multiple times
            self.CC = np.load(self.red_matrices_folder + "red_A_pd.npy")
        if not self.CL.size: # avoid loading multiple times
            self.CL = np.load(self.red_matrices_folder + "red_S.npy")
        if not self.LL.size: # avoid loading multiple times
            self.LL = np.load(self.red_matrices_folder + "red_F_d.npy")
    
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
        
        # Possibly copy the inner product matrix, if the primal problem has redefined id
        self.S = self.primal_problem.S
        
        # 9. I/O
        self.snap_folder = "snapshots__dual/"
        self.basis_folder = "basis__dual/"
        self.dual_folder = "dual__dual/"
        self.red_matrices_folder = "red_matr__dual/"
        self.pp_folder = "pp__dual/" # post processing
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    # Propagate the values of all setters also to the primal problem
    
    def setNmax(self, nmax):
        EllipticCoerciveRBBase.setNmax(self, nmax)
        self.primal_problem.setNmax(nmax)
    def settol(self, tol):
        EllipticCoerciveRBBase.settol(self, tol)
        self.primal_problem.settol(tol)
    def setmu_range(self, mu_range):
        EllipticCoerciveRBBase.setmu_range(self, mu_range)
        self.primal_problem.setmu_range(mu_range)
    def setxi_train(self, ntrain, sampling="random"):
        EllipticCoerciveRBBase.setxi_train(self, ntrain, sampling)
        self.primal_problem.setxi_train(ntrain, sampling)
    def setmu(self, mu):
        EllipticCoerciveRBBase.setmu(self, mu)
        self.primal_problem.setmu(mu)
        
    #  @}
    ########################### end - SETTERS - end ########################### 
        
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return the alpha_lower bound.
    def get_alpha_lb(self):
        return self.primal_problem.get_alpha_lb()
    
    ## Set theta multiplicative terms of the affine expansion of a.
    def compute_theta_a(self):
        return self.primal_problem.compute_theta_a()
    
    ## Set theta multiplicative terms of the affine expansion of f.
    def compute_theta_f(self):
        primal_theta_s = self.primal_problem.compute_theta_s()
        for qs in range(primal_theta_s):
            primal_theta_s[qs] *= -1.
        return (self.mu[1],)
    
    ## Set matrices resulting from the truth discretization of a.
    def assemble_truth_a(self):
        primal_truth_a = self.primal_problem.assemble_truth_a()
        primal_truth_a_transpose = ()
        for qa in range(primal_theta_a):
            primal_truth_a_transpose += (self.compute_transpose(primal_truth_a[qa]),)
        return primal_truth_a_transpose
    
    ## Set vectors resulting from the truth discretization of f.
    def assemble_truth_f(self):
        return self.primal_problem.assemble_truth_s()
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
