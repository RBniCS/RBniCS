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
## @file elliptic_coercive_base.py
#  @brief Implementation of projection based reduced order models for parabolic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from elliptic_coercive_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARABOLIC COERCIVE BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParablicCoerciveBase
#
# Base class containing the interface of a projection based ROM
# for parabolic coercive problems
class ParabolicCoerciveBase(EllipticCoerciveBase):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V):
        # Call the parent initialization
        EllipticCoerciveBase.__init__(self, V)
        
        # $$ PROBLEM SPECIFIC $$ #
        # Time step
        self.dt = 0.01
        # Final time
        self.T = 1.
        # All times
        self.all_times = np.linspace(self.dt, self.T, num = ceil(self.T/self.dt))
        # Current time
        self.t = 0.

        # $$ ONLINE DATA STRUCTURES $$ #
        # 3a. Number of terms in the affine expansion
        self.Qm = 1
        # 3b. Theta multiplicative factors of the affine expansion
        self.theta_m = (1., )
        # 3c. Reduced order matrices/vectors
        self.red_M = []
        # 4. Online solution
        self.uN = np.array([]) # array (size of T/dt + 1) of vectors of dimension N storing the reduced order solution
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 3c. Matrices/vectors resulting from the truth discretization
        u = self.u
        v = self.v
        truth_m = inner(u,v)*dx
        self.truth_M = (assemble(truth_m), )
        # 4. Auxiliary functions
        self.snap = np.array([]) # array (size of T/dt + 1) of vectors for storage of a truth solution
        self.red = np.array([]) # array (size of T/dt + 1) of vectors for storage of the FE reconstruction of the reduced solution
        self.er = np.array([]) # array (size of T/dt + 1) of vectors for storage of the error
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Perform an online solve. self.N will be used as matrix dimension if the default value is provided for N.
    def online_solve(self, mu, N=None, with_plot=True):
        if N is None:
            N = self.N
        self.load_red_matrices()
        self.setmu(mu)
        
        # Set the initial condition
        self.t = 0.
        self.uN = np.array.zeros([N, 1]) # as column vector
        
        for t in self.all_times:
            self.t = t
            self.red_solve(N)
        
        # Now obtain the FE functions corresponding to the reduced order solutions
        for k in range(1 + len(self.all_times)):
            sol = self.Z[:, 0]*self.uN[0, k]
            i=1
            for un in self.uN[1:, k]:
                sol += self.Z[:, i]*un
                i+=1
            self.red = np.hstack((self.red, sol.vector())) # add new solutions as column vectors
            if with_plot == True:
                plot(self.red[:, k], title = "Reduced solution. mu = " + str(self.mu) + ", t = " + str(k*dt), interactive = True)
    
    # Perform an online solve (internal)
    def red_solve(self, N):
        end = self.uN.shape[1]-1
        # Need to possibly re-evaluate theta, since they may be time dependent
        self.theta_a = self.compute_theta_a()
        self.theta_f = self.compute_theta_f()
        assembled_red_M = self.aff_assemble_red(self.red_M, self.theta_m, N, N)
        assembled_red_A = self.aff_assemble_red(self.red_A, self.theta_a, N, N)
        assembled_red_F = self.aff_assemble_red(self.red_F, self.theta_f, N, 1)
        assembled_red_lhs = 1./dt*assembled_red_M + assembled_red_A
        assembled_red_rhs = 1./dt*assembled_red_M*self.uN[:,end] + assembled_red_F
        if isinstance(assembled_red_lhs, float) == True:
            current_uN = assembled_red_rhs/assembled_red_lhs
        else:
            current_uN = np.linalg.solve(assembled_red_lhs, assembled_red_rhs)
        self.uN = np.hstack((self.uN, current_uN)) # add new solutions as column vectors
    
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{

    ## Perform a truth solve
    def truth_solve(self):
        current_snap = Function(self.V)
        
        # Set the initial condition
        self.t = 0.
        current_snap *= 0.
        self.snap = np.array(ic.vector()).reshape(-1, 1) # as column vector
        
        for k in len(self.all_times):
            self.t = self.all_times[k]
            # Need to possibly re-evaluate theta, since they may be time dependent
            self.theta_a = self.compute_theta_a()
            self.theta_f = self.compute_theta_f()
            assembled_truth_M = self.aff_assemble_truth(self.truth_M, self.theta_m)
            assembled_truth_A = self.aff_assemble_truth(self.truth_A, self.theta_a)
            assembled_truth_F = self.aff_assemble_truth(self.truth_F, self.theta_f)
            assembled_truth_lhs = 1./dt*assembled_truth_M + assembled_truth_A
            assembled_truth_rhs = 1./dt*assembled_truth_M*self.snap[:, k-1] + assembled_truth_F
            solve(assembled_truth_lhs, current_snap.vector(), assembled_truth_rhs)
            self.snap = np.hstack((self.uN, current_snap)) # add new solutions as column vectors
        
    ## Assemble the reduced order affine expansion (matrix)
    def build_red_matrices(self):
        # Assemble the reduced matrix A, as in parent
        EllipticCoerciveBase.build_red_matrices()
        # Moreover, assemble also the reduced matrix M
        dim = self.dim
        red_M = ()
        i = 0
        for M in self.truth_M:
            M = as_backend_type(M)
            if self.N == 1:
                red_M += (np.dot(self.Z.T,M.mat().getValues(range(dim),range(dim)).dot(self.Z)),)
            else:
                red = np.matrix(np.dot(self.Z.T,np.matrix(np.dot(M.mat().getValues(range(dim),range(dim)),self.Z))))
                red_M += (red,)
                i += 1
        self.red_M = red_M
        np.save(self.red_matrices_folder + "red_M", self.red_M)
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Load reduced order data structures
    def load_red_matrices(self):
        # Read in data structures as in parent
        EllipticCoerciveBase.load_red_matrices()
        # Moreover, read in also the reduced matrix M
        if not self.red_M: # avoid loading multiple times
            self.red_M = np.load(self.red_matrices_folder + "red_M.npy")
    
    #  @}
    ########################### end - I/O - end ########################### 
