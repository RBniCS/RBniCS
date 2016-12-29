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

from RBniCS.problems.base import ParametrizedDifferentialProblem
from RBniCS.backends import LinearSolver, product, sum, transpose
from RBniCS.utils.decorators import Extends, override

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE PROBLEM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveProblem
#
# Base class containing the definition of elliptic coercive problems
@Extends(ParametrizedDifferentialProblem)
class EllipticCoerciveProblem(ParametrizedDifferentialProblem):
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
    @override
    def __init__(self, V, **kwargs):
        # Call to parent
        ParametrizedDifferentialProblem.__init__(self, V, **kwargs)
        
        # Form names for elliptic problems
        self.terms = ["a", "f"]
        self.terms_order = {"a": 2, "f": 1}
        self.components = ["u"]
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform a truth solve
    @override
    def solve(self, **kwargs):
        assembled_operator = dict()
        assembled_operator["a"] = sum(product(self.compute_theta("a"), self.operator["a"]))
        assembled_operator["f"] = sum(product(self.compute_theta("f"), self.operator["f"]))
        if self.dirichlet_bc is not None:
            assembled_dirichlet_bc = sum(product(self.compute_theta("dirichlet_bc"), self.dirichlet_bc))
        else:
            assembled_dirichlet_bc = None
        solver = LinearSolver(assembled_operator["a"], self._solution, assembled_operator["f"], assembled_dirichlet_bc)
        solver.solve()
        return self._solution
        
    ## Perform a truth evaluation of the (compliant) output
    @override
    def output(self):
        assembled_output_operator = sum(product(self.compute_theta("f"), self.operator["f"]))
        self._output = transpose(assembled_output_operator)*self._solution
        return self._output
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
