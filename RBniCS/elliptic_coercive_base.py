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
## @file elliptic_coercive_base.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
import numpy as np
import sys # for exit
import itertools # for equispaced grid generation
from parametrized_problem import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems
class EllipticCoerciveBase(ParametrizedProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, bc_list):
        # Call to parent
        ParametrizedProblem.__init__(self)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # 3a. Number of terms in the affine expansion
        self.Qa = 0
        self.Qf = 0
        # 3b. Theta multiplicative factors of the affine expansion
        self.theta_a = ()
        self.theta_f = ()
        # 3c. Reduced order matrices/vectors
        self.reduced_A = ()
        self.reduced_F = ()
        # 4. Online solution
        self.uN = 0 # vector of dimension N storing the reduced order solution
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 3c. Matrices/vectors resulting from the truth discretization
        self.truth_A = ()
        self.truth_F = ()
        # 4. Offline solutions
        self.snapshot = Function(V) # temporary vector for storage of a truth solution
        self.reduced = Function(V) # temporary vector for storage of the FE reconstruction of the reduced solution
        self.error = Function(V) # temporary vector for storage of the error
        # 6. Basis functions matrix
        self.Z = np.array([])
        # 7. Truth space, functions and inner products
        self.bc_list = bc_list
        self.V = V
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        u = self.u
        v = self.v
        scalar = inner(u,v)*dx + inner(grad(u),grad(v))*dx # H^1 inner product
        self.S = assemble(scalar) # H^1 inner product matrix
        if self.bc_list != None:
            [bc.apply(self.S) for bc in self.bc_list] # make sure to apply BCs to the inner product matrix
    
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Perform an online solve. self.N will be used as matrix dimension if the default value is provided for N.
    def online_solve(self, N=None, with_plot=True):
        self.load_reduced_matrices()
        if N is None:
            N = self.N
        self._online_solve(N)
        sol = self.Z[:, 0]*self.uN[0]
        i=1
        for un in self.uN[1:]:
            sol += self.Z[:, i]*un
            i+=1
        self.reduced.vector()[:] = sol
        if with_plot == True:
            self._plot(self.reduced, title = "Reduced solution. mu = " + str(self.mu), interactive = True)
    
    # Perform an online solve (internal)
    def _online_solve(self, N):
        self.theta_a = self.compute_theta_a()
        self.theta_f = self.compute_theta_f()
        assembled_reduced_A = self.affine_assemble_reduced_matrix(self.reduced_A, self.theta_a, N, N)
        assembled_reduced_F = self.affine_assemble_reduced_vector(self.reduced_F, self.theta_f, N)
        if isinstance(assembled_reduced_A, float) == True:
            uN = assembled_reduced_F/assembled_reduced_A
        else:
            uN = np.linalg.solve(assembled_reduced_A, assembled_reduced_F)
            uN = uN.reshape(-1, 1) # as column vector
        self.uN = uN
        
    ## Assemble the reduced affine expansion (matrix)
    def affine_assemble_reduced_matrix(self, vec, theta_v, m, n):
        A_ = vec[0][:m,:n]*theta_v[0]
        assert len(vec) == len(theta_v)
        for i in range(1,len(vec)):
            A_ += vec[i][:m,:n]*theta_v[i]
        return A_
        
    ## Assemble the reduced affine expansion (vector)
    def affine_assemble_reduced_vector(self, vec, theta_v, n):
        F_ = vec[0][:n]*theta_v[0]
        assert len(vec) == len(theta_v)
        for i in range(1,len(vec)):
            F_ += vec[i][:n]*theta_v[i]
        return F_
    
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
    def offline(self):
        sys.exit("Please implement the offline phase of the reduced order model.")

    ## Perform a truth solve
    def truth_solve(self):
        self.theta_a = self.compute_theta_a()
        self.theta_f = self.compute_theta_f()
        assembled_truth_A = self.affine_assemble_truth_matrix(self.truth_A, self.theta_a)
        assembled_truth_F = self.affine_assemble_truth_vector(self.truth_F, self.theta_f)
        solve(assembled_truth_A, self.snapshot.vector(), assembled_truth_F)
        
    ## Assemble the truth affine expansion (matrix)
    def affine_assemble_truth_matrix(self, vec, theta_v):
        A_ = vec[0]*theta_v[0]
        for i in range(1,len(vec)):
            A_ += vec[i]*theta_v[i]
        return A_
        
    ## Assemble the symmetric part of the the truth affine expansion (matrix)
    def affine_assemble_truth_symmetric_part_matrix(self, vec, theta_v):
        A_ = self.affine_assemble_truth_matrix(vec, theta_v)
        AT_ = self.compute_transpose(A_)
        A_ += AT_
        A_ /= 2.
        return A_
        
    ## Assemble the truth affine expansion (vector)
    #  (the implementation is acutally the same of the matrix case, but this method is
    #   provided here for symmetry with the reduced case)
    def affine_assemble_truth_vector(self, vec, theta_v):
        F_ = vec[0]*theta_v[0]
        for i in range(1,len(vec)):
            F_ += vec[i]*theta_v[i]
        return F_
        
    ## Apply BCs to each element of the truth affine expansion (matrix)
    def apply_bc_to_matrix_expansion(self, vec):
        if self.bc_list != None:
            for i in range(len(vec)):
                [bc.apply(vec[i]) for bc in self.bc_list]
            for i in range(1,len(vec)):
                [bc.zero(vec[i]) for bc in self.bc_list]
            
    ## Apply BCs to each element of the truth affine expansion (vector)
    def apply_bc_to_vector_expansion(self, vec):
        if self.bc_list != None:
            for i in range(len(vec)):
                [bc.apply(vec[i]) for bc in self.bc_list]
        
    ## Assemble the reduced order affine expansion (matrix)
    def build_reduced_matrices(self):
        reduced_A = ()
        for A in self.truth_A:
            A = as_backend_type(A)
            dim = A.size(0) # = A.size(1)
            if self.N == 1:
                reduced_A += (np.dot(self.Z.T,A.mat().getValues(range(dim),range(dim)).dot(self.Z)),)
            else:
                reduced_A += (np.matrix(np.dot(self.Z.T,np.matrix(np.dot(A.mat().getValues(range(dim),range(dim)),self.Z)))),)
        self.reduced_A = reduced_A
        np.save(self.reduced_matrices_folder + "reduced_A", self.reduced_A)
    
    ## Assemble the reduced order affine expansion (rhs)
    def build_reduced_vectors(self):
        reduced_F = ()
        for F in self.truth_F:
            F = as_backend_type(F)
            dim = F.size()
            reduced_f = np.dot(self.Z.T, F.vec().getValues(range(dim)) )
            reduced_F += (reduced_f,)
        self.reduced_F = reduced_F
        np.save(self.reduced_matrices_folder + "reduced_F", self.reduced_F)
    
    ## Auxiliary internal method to compute the scalar product (v1, M*v2)
    def compute_scalar(self,v1,v2,M):
        return v1.vector().inner(M*v2.vector())
        
    ## Auxiliary internal method to compute the transpose of a matrix
    def compute_transpose(self, A):
        AT = A.copy()
        A = as_backend_type(A)
        AT = as_backend_type(AT)
        A.mat().transpose(AT.mat())
        return AT
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu
    def compute_error(self, N=None, skip_truth_solve=False):
        if not skip_truth_solve:
            self.truth_solve()
        self.online_solve(N, False)
        self.error.vector()[:] = self.snapshot.vector()[:] - self.reduced.vector()[:] # error as a function
        self.theta_a = self.compute_theta_a() # not really necessary, for symmetry with the parabolic case
        assembled_truth_A_sym = self.affine_assemble_truth_symmetric_part_matrix(self.truth_A, self.theta_a)
        error_norm_squared = self.compute_scalar(self.error, self.error, assembled_truth_A_sym) # norm of the error
        return np.sqrt(error_norm_squared)
        
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    def error_analysis(self, N=None):
        sys.exit("Please implement the error analysis of the reduced order model.")
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Load reduced order data structures
    def load_reduced_matrices(self):
        if len(np.asarray(self.reduced_A)) == 0: # avoid loading multiple times
            self.reduced_A = tuple(np.load(self.reduced_matrices_folder + "reduced_A.npy"))
        if len(np.asarray(self.reduced_F)) == 0: # avoid loading multiple times
            self.reduced_F = tuple(np.load(self.reduced_matrices_folder + "reduced_F.npy"))
        if len(np.asarray(self.Z)) == 0: # avoid loading multiple times
            self.Z = np.load(self.basis_folder + "basis.npy")
            self.N = self.Z.shape[1]

    ## Export solution in VTK format
    def export_solution(self, solution, filename):
        self._export_vtk(solution, filename, {"With mesh motion": True, "With preprocessing": True})
        
    ## Export basis in VTK format. 
    def export_basis(self, basis, filename):
        self._export_vtk(basis, filename, {"With mesh motion": False, "With preprocessing": False})
        
    #  @}
    ########################### end - I/O - end ########################### 

    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{

    ## Return theta multiplicative terms of the affine expansion of a.
    # example of implementation:
    #    m1 = self.mu[0]
    #    m2 = self.mu[1]
    #    m3 = self.mu[2]
    #    theta_a0 = m1
    #    theta_a1 = m2
    #    theta_a2 = m1*m2+m3/7.0
    #    return (theta_a0, theta_a1, theta_a2)
    def compute_theta_a(self):
        print "The function compute_theta_a() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function compute_theta_a(self)!")
    
    ## Return theta multiplicative terms of the affine expansion of f.
    # example of implementation:
    #    m1 = self.mu[0]
    #    m2 = self.mu[1]
    #    m3 = self.mu[2]
    #    theta_f0 = m1
    #    theta_f1 = m2
    #    theta_f2 = m1*m2+m3/7.0
    #    return (theta_f0, theta_f1, theta_f2)
    def compute_theta_f(self):
        print "The function compute_theta_f() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function compute_theta_f(self)!")
        
    ## Return matrices resulting from the truth discretization of a.
    # example of implementation:
    #    a0 = inner(grad(u),grad(v))*dx
    #    A0 = assemble(a0)
    #    return (A0,)
    def assemble_truth_a(self):
        print "The function assemble_truth_a() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function assemble_truth_a(self)!")

    ## Return vectors resulting from the truth discretization of f.
    #    f0 = v*ds(1)
    #    F0 = assemble(f0)
    #    return (F0,)
    def assemble_truth_f(self):
        print "The function compute_truth_f() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function assemble_truth_f(self)!")
    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

