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

from dolfin import Function
from RBniCS.problems.parametrized_problem import ParametrizedProblem
from RBniCS.linear_algebra import AffineExpansionOfflineStorage, product, transpose, solve, sum

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
    def __init__(self, V, **kwargs):
        # Call to parent
        ParametrizedProblem.__init__(self, self.name())
        
        # Input arguments
        self.V = V
        # Number of terms in the affine expansion
        self.terms = ["a", "f"]
        self.Q = dict() # from string to integer
        # Matrices/vectors resulting from the truth discretization
        self.operator = dict() # from string to AffineExpansionOfflineStorage
        self.inner_product = AffineExpansionOfflineStorage() # even though it will contain only one matrix
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
        for term in self.terms:
            self.operator[term] = AffineExpansionOfflineStorage(self.assemble_operator(term))
            self.Q[term] = len(self.operator[term])
        self.inner_product.init(self.assemble_operator("inner_product"))
        try:
            self.dirichlet_bc.init(self.assemble_operator("dirichlet_bc"))
        except RuntimeError: # there were no Dirichlet BCs
            pass
                    
    ## Perform a truth solve
    def solve(self):
        assembled_operator = dict()
        for term in self.terms:
            assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term]))
        if len(self.dirichlet_bc) > 0:
            try:
                theta_dirichlet_bc = self.compute_theta("dirichlet_bc")
            except RuntimeError: # there were no theta functions
                # We provide in this case a shortcut for the case of homogeneous Dirichlet BCs,
                # that do not require an additional lifting functions.
                # The user needs to implement the dirichlet_bc case for assemble_operator, 
                # but not the one in compute_theta (since theta would not matter, being multiplied by zero)
                theta_dirichlet_bc = (0,)*len(self.dirichlet_bc)
                
            assembled_dirichlet_bc = sum(product(theta_dirichlet_bc, self.dirichlet_bc))
        else:
            assembled_dirichlet_bc = None
        solve(assembled_operator["a"], self._solution.vector(), assembled_operator["f"], assembled_dirichlet_bc)
        return self._solution
        
    ## Perform a truth evaluation of the (compliant) output
    def output(self):
        assembled_output_operator = sum(product(self.compute_theta("f"), self.operator["f"]))
        self._output = transpose(assembled_output_operator)*self._solution.vector()
        return self._output
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Export solution in VTK format
    def export_solution(self, solution, folder, filename):
        self._export_vtk(solution, folder, filename, with_mesh_motion=True, with_preprocessing=True)
        
    ## Get the name of the problem, to be used as a prefix for output folders
    @classmethod
    def name(cls):
        return cls.__name__
        
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
        raise RuntimeError("The method compute_theta() is problem-specific and needs to be overridden.")
        
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
        raise RuntimeError("The method assemble_operator() is problem-specific and needs to be overridden.")
        
    ## Return a lower bound for the coercivity constant
    # Example of implementation:
    #    return 1.0
    # Note that this method is not needed in POD-Galerkin reduced order models.
    def get_stability_factor(self):
        raise RuntimeError("The method get_stability_factor() is problem-specific and needs to be overridden.")
    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

