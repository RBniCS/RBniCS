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
## @file elliptic_coercive_reduced_problem.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from math import sqrt
from numpy import isclose
from RBniCS.problems.base import ParametrizedReducedDifferentialProblem
from RBniCS.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from RBniCS.backends import LinearSolver, product, sum, transpose
from RBniCS.backends.online import OnlineFunction
from RBniCS.utils.decorators import Extends, override, ReducedProblemFor, MultiLevelReducedProblem
from RBniCS.reduction_methods.elliptic_coercive import EllipticCoerciveReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveReducedOrderModelBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(ParametrizedReducedDifferentialProblem) # needs to be first in order to override for last the methods.
@ReducedProblemFor(EllipticCoerciveProblem, EllipticCoerciveReductionMethod)
@MultiLevelReducedProblem
class EllipticCoerciveReducedProblem(ParametrizedReducedDifferentialProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        ParametrizedReducedDifferentialProblem.__init__(self, truth_problem, **kwargs)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
            
    # Perform an online solve. self.N will be used as matrix dimension if the default value is provided for N.
    @override
    def solve(self, N=None, **kwargs):
        if N is None:
            N = self.N
        uN = self._solve(N, **kwargs)
        return uN
    
    # Perform an online solve (internal)
    def _solve(self, N, **kwargs):
        N += self.N_bc
        assembled_operator = dict()
        assembled_operator["a"] = sum(product(self.compute_theta("a"), self.operator["a"][:N, :N]))
        assembled_operator["f"] = sum(product(self.compute_theta("f"), self.operator["f"][:N]))
        if self.dirichlet_bc and not self.dirichlet_bc_are_homogeneous:
            theta_bc = self.compute_theta("dirichlet_bc")
        else:
            theta_bc = None
        self._solution = OnlineFunction(N)
        solver = LinearSolver(assembled_operator["a"], self._solution, assembled_operator["f"], theta_bc)
        solver.solve()
        return self._solution
        
    # Perform an online evaluation of the (compliant) output
    @override
    def output(self):
        N = self._solution.N
        assembled_output_operator = sum(product(self.compute_theta("f"), self.operator["f"][:N]))
        self._output = transpose(assembled_output_operator)*self._solution
        return self._output
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu
    @override
    def compute_error(self, N=None, **kwargs):
        if self._compute_error__previous_mu != self.mu:
            self.truth_problem.solve(**kwargs)
            self.truth_problem.output()
            # Do not carry out truth solves anymore for the same parameter
            self._compute_error__previous_mu = self.mu
        # Compute the error on the solution and output
        self.solve(N, **kwargs)
        self.output()
        return self._compute_error()
        
    # Internal method for error computation
    def _compute_error(self):
        N = self._solution.N
        # Compute the error on the solution
        reduced_solution = self.Z[:N]*self._solution
        truth_solution = self.truth_problem._solution
        error = truth_solution - reduced_solution
        assembled_error_inner_product_operator = sum(product(self.truth_problem.compute_theta("a"), self.truth_problem.operator["a"])) # use the energy norm (skew part will discarded by the scalar product)
        error_norm_squared = transpose(error)*assembled_error_inner_product_operator*error # norm SQUARED of the error
        # Compute the error on the output
        error_output = abs(self.truth_problem._output - self._output)
        assert error_norm_squared >= 0. or isclose(error_norm_squared, 0.)
        return (sqrt(abs(error_norm_squared)), error_output)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ###########################

