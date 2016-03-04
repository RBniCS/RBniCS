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
## @file elliptic_coercive_problem.py
#  @brief Base class for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.parametrized_problem import ParametrizedProblem

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE PROBLEM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveProblem
#
# Base class containing the definition of elliptic coercive problems
class EllipticCoerciveProblem(ParametrizedProblem):
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
    ## @defgroup Constructors Methods related to the construction of the elliptic problem
    #  @{
    
    ## Default initialization of members
    def __init__(self, V):
        # Call to parent
        ParametrizedProblem.__init__(self)
        
        # Input arguments
        self.V = V
        # 3a. Number of terms in the affine expansion
        self.Qa = 0
        self.Qf = 0
        # 3b. Theta multiplicative factors of the affine expansion
        self.theta_a = tuple()
        self.theta_f = tuple()
        # 3c. Matrices/vectors resulting from the truth discretization
        self.operator_a = AffineExpansionOfflineStorage()
        self.operator_f = AffineExpansionOfflineStorage()
        self.inner_product = AffineExpansionOfflineStorage()
        self.dirichlet_bc = AffineExpansionOfflineStorage()
        # Solution
        self._solution = Function(self.V)
        self._output = 0
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Initialize data structures required for the offline phase
    def init(self):
        self.operator_a = AffineExpansionOfflineStorage(self.assemble_operator("a"))
        self.operator_f = AffineExpansionOfflineStorage(self.assemble_operator("f"))
        self.inner_product = AffineExpansionOfflineStorage(self.assemble_operator("inner_product"))
        try:
            self.dirichlet_bc = AffineExpansionOfflineStorage(self.assemble_operator("dirichlet_bc"))
        except RuntimeError: # there were no Dirichlet BCs
            self.dirichlet_bc = AffineExpansionOfflineStorage()
        self.Qa = len(self.operator_a)
        self.Qf = len(self.operator_f)
        
    ## Perform a truth solve
    def solve(self):
        self.theta_a = self.compute_theta("a")
        self.theta_f = self.compute_theta("f")
        try:
            theta_bc = self.compute_theta("dirichlet_bc")
        except RuntimeError: # there were no Dirichlet BCs
            theta_bc = ()
        assembled_operator_a = sum(product(self.theta_a, self.operator_a))
        assembled_operator_f = sum(product(self.theta_f, self.operator_f))
        assembled_dirichlet_bc = sum(product(theta_bc, self.dirichet_bc))
        solve(assembled_operator_a == assembled_operator_f, self._solution, assembled_dirichlet_bc)
        return self._solution
        
    ## Perform a truth evaluation of the (compliant) output
    def output(self):
        self.theta_f = self.compute_theta("f")
        assembled_operator_f = sum(product(self.theta_f, self.operator_f))
        self._output = transpose(assembled_operator_f)*self._solution.vector()
        return self._output
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Export solution in VTK format
    def export_solution(self, solution, filename):
        self._export_vtk(solution, filename, with_mesh_motion=True, with_preprocessing=True)
        
    #  @}
    ########################### end - I/O - end ########################### 

    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{

    ## Return theta multiplicative terms of the affine expansion of the problem.
    # Example of implementation:
    #   m1 = self.mu[0]
    #   m2 = self.mu[1]
    #   m3 = self.mu[2]
    #   if term == "a":
    #       theta_a0 = m1
    #       theta_a1 = m2
    #       theta_a2 = m1*m2+m3/7.0
    #       return (theta_a0, theta_a1, theta_a2)
    #   elif term == "f":
    #       theta_f0 = m1*m3
    #       return (theta_f0,)
    #   elif term == "dirichlet_bc":
    #       theta_bc0 = 1.
    #       return (theta_f0,)
    #   else:
    #       raise RuntimeError("Invalid term for compute_theta().")
    def compute_theta(self, term):
        raise RuntimeError("The function compute_theta() is problem-specific and needs to be overridden.")
        
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    # Example of implementation:
    #   if term == "a":
    #       a0 = inner(grad(u),grad(v))*dx
    #       return (a0,)
    #   elif term == "f":
    #       f0 = v*ds(1)
    #       return (f0,)
    #   elif term == "dirichlet_bc":
    #       bc0 = [(V, Constant(0.0), boundaries, 3)]
    #       return (bc0,)
    #   elif term == "inner_product":
    #       x0 = u*v*dx + inner(grad(u),grad(v))*dx
    #       return (x0,)
    #   else:
    #       raise RuntimeError("Invalid term for assemble_operator().")
    def assemble_operator(self, term):
        raise RuntimeError("The function assemble_operator() is problem-specific and needs to be overridden.")
        
    ## Return a lower bound for the coercivity constant
    # example of implementation:
    #    return 1.0
    # Note that this method is not needed in POD-Galerkin reduced order models.
    def get_alpha_lb(self):
        raise RuntimeError("The function get_alpha_lb(self) is problem-specific and needs to be overridden.")
    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

