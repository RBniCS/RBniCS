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
## @file elliptic_coercive_base.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.parametrized_problem import ParametrizedProblem

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems
class EllipticCoerciveBase(ParametrizedProblem):
    """This class defines and implement variables and methods needed for
    solving an elliptic and coercive problem. This class specializes
    in the two currently implemented reduced order methods, namely the
    Reduced Basis Method (EllipticCoerciveRBBase), and the Proper
    Orthogonal Decomposition (EllipticCoercivePODBase). These two
    classes assume that the output(s) of interest is (are)
    compliant. Whether the compliancy hypothesis does not hold, the
    EllipticCoerciveRBNonCompliantBase must be used.

    In particular, this class implements the following functions, whose name are self-explanatory:

    ## Methods related to the offline stage
    - offline() # to be overridden 
    - truth_solve()
    - affine_assemble_truth_matrix()
    - affine_assemble_truth_symmetric_part_matrix()
    - affine_assemble_truth_vector()
    - apply_bc_to_matrix_expansion()
    - apply_bc_to_vector_expansion()
    - build_reduced_matrices()
    - build_reduced_vectors()
    - compute_scalar()
    - compute_transpose()

    ## Methods related to the online stage
    - online_solve()
    - affine_assemble_reduced_matrix()
    - affine_assemble_reduced_vector()

    ## Error analysis
    - compute_error()
    - error_analysis() # to be overridden

    ## Input/output methods
    - load_reduced_matrices()
    - export_solution()
    - export_basis()

    ## Problem specific methods
    - compute_theta_a() # to be overridden
    - compute_theta_f() # to be overridden
    - assemble_truth_a() # to be overridden
    - assemble_truth_f() # to be overridden

    If you want/need to implement an alternate reduced order method,
    (e.g., CVT), you might want to derive from this class.

    """
    
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
        self.theta_a = tuple()
        self.theta_f = tuple()
        # 3c. Reduced order matrices/vectors
        self.reduced_A = AffineExpansionOnlineStorage()
        self.reduced_F = AffineExpansionOnlineStorage()
        # 4. Online solution
        self.uN = OnlineVector() # vector of dimension N storing the reduced order solution
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 3c. Matrices/vectors resulting from the truth discretization
        self.truth_A = tuple()
        self.truth_F = tuple()
        # 4. Offline solutions
        self.snapshot = Function(V) # temporary vector for storage of a truth solution
        self.reduced = Function(V) # temporary vector for storage of the FE reconstruction of the reduced solution
        self.error = Function(V) # temporary vector for storage of the error
        # 6. Basis functions matrix
        self.Z = BasisFunctionsMatrix()
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
        self.reduced.vector() = self.Z*self.uN
        if with_plot == True:
            self._plot(self.reduced, title = "Reduced solution. mu = " + str(self.mu), interactive = True)
    
    # Perform an online solve (internal)
    def _online_solve(self, N):
        self.theta_a = self.compute_theta_a()
        self.theta_f = self.compute_theta_f()
        assembled_reduced_A = sum(product(self.theta_a, self.reduced_A[:N, :N]))
        assembled_reduced_F = sum(product(self.theta_f, self.reduced_F[:N]))
        solve(assembled_reduced_A, self.uN, assembled_reduced_F)
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Initialize data structures required for the offline phase
    def _init_offline(self):
        self.truth_A = tuple(assemble(a_form) for a_form in self.assemble_truth_a())
        self.apply_bc_to_matrix_expansion(self.truth_A)
        self.truth_F = tuple(assemble(f_form) for f_form in self.assemble_truth_f())
        self.apply_bc_to_vector_expansion(self.truth_F)
        self.Qa = len(self.truth_A)
        self.Qf = len(self.truth_F)
        self.reduced_A = AffineExpansionOnlineStorage(self.Qa)
        self.reduced_F = AffineExpansionOnlineStorage(self.Qf)
        
    ## Perform a truth solve
    def truth_solve(self):
        self.theta_a = self.compute_theta_a()
        self.theta_f = self.compute_theta_f()
        assembled_truth_A = sum(product(self.theta_a, self.truth_A))
        assembled_truth_F = sum(product(self.theta_f, self.truth_F))
        solve(assembled_truth_A, self.snapshot.vector(), assembled_truth_F)
        
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
        for qa in range(self.Qa):
            self.reduced_A[qa] = transpose(self.Z)*self.truth_A[qa]*self.Z
        self.reduced_A.save(self.reduced_matrices_folder, "reduced_A")
    
    ## Assemble the reduced order affine expansion (rhs)
    def build_reduced_vectors(self):
        for qf in range(self.Qf):
            self.reduced_F[qf] = transpose(self.Z)*self.truth_F[qf]
        self.reduced_F.save(self.reduced_matrices_folder, "reduced_F")
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    ## Initialize data structures required for the error analysis phase
    def _init_error_analysis(self):
        self.truth_A = tuple(assemble(a_form) for a_form in self.assemble_truth_a())
        self.apply_bc_to_matrix_expansion(self.truth_A)
        self.truth_F = tuple(assemble(f_form) for f_form in self.assemble_truth_f())
        self.apply_bc_to_vector_expansion(self.truth_F)
        self.Qa = len(self.truth_A)
        self.Qf = len(self.truth_F)
        self.load_reduced_matrices()
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu
    def compute_error(self, N=None, skip_truth_solve=False):
        if not skip_truth_solve:
            self.truth_solve()
        self.online_solve(N, False)
        self.error.vector()[:] = self.snapshot.vector()[:] - self.reduced.vector()[:] # error as a function
        self.theta_a = self.compute_theta_a() # not really necessary, for symmetry with the parabolic case
        assembled_truth_A = sum(product(self.theta_a, self.truth_A)) # use the energy norm (skew part will discarded by the scalar product)
        error_norm_squared = self.compute_scalar_product(self.error, assembled_truth_A, self.error) # norm of the error
        return sqrt(error_norm_squared)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Load reduced order data structures
    def load_reduced_matrices(self):
        self.reduced_A.load(self.reduced_matrices_folder, "reduced_A")
        self.reduced_F.load(self.reduced_matrices_folder, "reduced_F")
        was_Z_loaded = self.Z.load(self.basis_folder, "basis")
        if was_Z_loaded:
            self.N = len(self.Z)

    ## Export solution in VTK format
    def export_solution(self, solution, filename):
        self._export_vtk(solution, filename, {"With mesh motion": True, "With preprocessing": True})
        
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
        raise RuntimeError("The function compute_theta_a() is problem-specific and needs to be overridden.")
    
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
        raise RuntimeError("The function compute_theta_f() is problem-specific and needs to be overridden.")
        
    ## Return matrices resulting from the truth discretization of a.
    # example of implementation:
    #    a0 = inner(grad(u),grad(v))*dx
    #    A0 = assemble(a0)
    #    return (A0,)
    def assemble_truth_a(self):
        raise RuntimeError("The function assemble_truth_a() is problem-specific and needs to be overridden.")

    ## Return vectors resulting from the truth discretization of f.
    #    f0 = v*ds(1)
    #    F0 = assemble(f0)
    #    return (F0,)
    def assemble_truth_f(self):
        raise RuntimeError("The function compute_truth_f() is problem-specific and needs to be overridden.")
    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

