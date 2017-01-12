# Copyright (C) 2015-2017 by the RBniCS authors
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
from RBniCS.problems.stokes.stokes_problem import StokesProblem
from RBniCS.backends import LinearSolver, product, sum, transpose
from RBniCS.backends.online import OnlineFunction
from RBniCS.utils.decorators import Extends, override, ReducedProblemFor, MultiLevelReducedProblem
from RBniCS.reduction_methods.stokes import StokesReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveReducedOrderModelBase
#
# Base class containing the interface of a projection based ROM
# for saddle point problems.
@Extends(ParametrizedReducedDifferentialProblem) # needs to be first in order to override for last the methods.
@ReducedProblemFor(StokesProblem, StokesReductionMethod)
@MultiLevelReducedProblem
class StokesReducedProblem(ParametrizedReducedDifferentialProblem):
    
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
        N, kwargs = self._online_size_from_kwargs(N, **kwargs)
        uN = self._solve(N, **kwargs)
        return uN
    
    # Perform an online solve (internal)
    def _solve(self, N, **kwargs):
        N += self.N_bc
        assembled_operator = dict()
        for term in ("a", "b", "bt", "f", "g"):
            assert self.terms_order[term] in (1, 2)
            if self.terms_order[term] == 2:
                assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term][:N, :N]))
            elif self.terms_order[term] == 1:
                assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term][:N]))
            else:
                raise AssertionError("Invalid value for order of term " + term)
        theta_bc = dict()
        for component in ("u", "p"):
            if self.dirichlet_bc[component] and not self.dirichlet_bc_are_homogeneous[component]:
                theta_bc[component] = self.compute_theta("dirichlet_bc_" + component)
        if len(theta_bc) == 0:
            theta_bc = None
        self._solution = OnlineFunction(N)
        solver = LinearSolver(
            assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"],
            self._solution,
            assembled_operator["f"] + assembled_operator["g"],
            theta_bc
        )
        solver.solve()
        return self._solution
        
    # Perform an online evaluation of the output
    @override
    def output(self):
        self._output = 1. # TODO
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
        velocity_inner_product = self.truth_problem.inner_product["u"][0]
        velocity_exact_norm_squared = transpose(truth_solution)*velocity_inner_product*truth_solution
        velocity_error_norm_squared = transpose(error)*velocity_inner_product*error
        pressure_inner_product = self.truth_problem.inner_product["p"][0]
        pressure_exact_norm_squared = transpose(truth_solution)*pressure_inner_product*truth_solution
        pressure_error_norm_squared = transpose(error)*pressure_inner_product*error
        # Compute the error on the output
        reduced_output = self._output
        truth_output = self.truth_problem._output
        error_output = abs(truth_output - reduced_output)
        assert velocity_error_norm_squared >= 0. or isclose(velocity_error_norm_squared, 0.)
        assert velocity_exact_norm_squared >= 0. or isclose(velocity_exact_norm_squared, 0.)
        assert pressure_error_norm_squared >= 0. or isclose(pressure_error_norm_squared, 0.)
        assert pressure_exact_norm_squared >= 0. or isclose(pressure_exact_norm_squared, 0.)
        return (sqrt(velocity_error_norm_squared), sqrt(velocity_error_norm_squared/velocity_exact_norm_squared), sqrt(pressure_error_norm_squared), sqrt(pressure_error_norm_squared/pressure_exact_norm_squared), error_output, error_output/truth_output)
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ###########################
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
        
    ## Assemble the reduced order affine expansion
    def assemble_operator(self, term, current_stage="online"):
        if term == "bt_restricted":
            self.operator["bt_restricted"] = self.operator["bt"]
            return self.operator["bt_restricted"]
        elif term == "inner_product_s":
            self.inner_product["s"] = self.inner_product["u"]
            return self.inner_product["s"]
        else:
            return ParametrizedReducedDifferentialProblem.assemble_operator(self, term, current_stage)
                                
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

