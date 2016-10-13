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
from RBniCS.problems.base import ParametrizedReducedDifferentialProblem
from RBniCS.problems.saddle_point.saddle_point_problem import SaddlePointProblem
from RBniCS.backends import difference, LinearSolver, product, sum, transpose
from RBniCS.backends.online import OnlineFunction
from RBniCS.utils.decorators import Extends, override, ReducedProblemFor, MultiLevelReducedProblem
from RBniCS.reduction_methods.saddle_point import SaddlePointReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveReducedOrderModelBase
#
# Base class containing the interface of a projection based ROM
# for saddle point problems.
@Extends(ParametrizedReducedDifferentialProblem) # needs to be first in order to override for last the methods.
@ReducedProblemFor(SaddlePointProblem, SaddlePointReductionMethod)
@MultiLevelReducedProblem
class SaddlePointReducedProblem(ParametrizedReducedDifferentialProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem):
        # Call to parent
        ParametrizedReducedDifferentialProblem.__init__(self, truth_problem)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        self.Z.init(self.truth_problem.component_name_to_basis_component_index, self.truth_problem.component_name_to_function_component)
        
        # Call Parent initialization
        ParametrizedReducedDifferentialProblem.init(self, current_stage)
            
    # Perform an online solve. self.N will be used as matrix dimension if the default value is provided for N.
    @override
    def solve(self, N=None, **kwargs):
        N, kwargs = self._online_size_from_kwargs(N, **kwargs)
        uN = self._solve(N, **kwargs)
        return uN
    
    # Perform an online solve (internal)
    def _solve(self, N, **kwargs):
        for component_name in ("u", "s", "p"):
            N[component_name] += self.N_bc[component_name]
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
        for component_name in ("u", "p"):
            if self.dirichlet_bc[component_name] and not self.dirichlet_bc_are_homogeneous[component_name]:
                theta_bc[component_name] = self.compute_theta("dirichlet_bc_" + component_name)
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
        
    def _online_size_from_kwargs(self, N, **kwargs):
        if N is None:
            for component_name in ("u", "s", "p"):
                if component_name not in kwargs:
                    N = dict(self.N) # copy the default dict
                    break
            else: # for loop was not broken: all components found
                N = {component_name:kwargs[component_name] for component_name in ("u", "s", "p")}
                for component_name in ("u", "s", "p"):
                    del kwargs[component_name]
        else:
            assert isinstance(N, int)
            N = {component_name:N for component_name in ("u", "s", "p")}
            for component_name in ("u", "s", "p"):
                assert component_name not in kwargs
        return N, kwargs
        
    # Perform an online evaluation of the output
    @override
    def output(self):
        return 0. # TODO
        
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
        return self._compute_error((self.truth_problem._solution, self.truth_problem._output), (self._solution, self._output))
        
    # Internal method for error computation
    def _compute_error(self, truth_solution_and_output, reduced_solution_and_output):
        N = self._solution.vector().N
        # Compute the error on the solution
        reduced_solution = self.Z[:N]*reduced_solution_and_output[0]
        truth_solution = truth_solution_and_output[0]
        error = difference(truth_solution, reduced_solution)
        velocity_inner_product = self.truth_problem.inner_product["u"][0]
        velocity_error_norm_squared = transpose(error.vector())*velocity_inner_product*error.vector()
        pressure_inner_product = self.truth_problem.inner_product["p"][0]
        pressure_error_norm_squared = transpose(error.vector())*pressure_inner_product*error.vector()
        # Compute the error on the output
        error_output = abs(truth_solution_and_output[1] - reduced_solution_and_output[1])
        return (sqrt(velocity_error_norm_squared), sqrt(pressure_error_norm_squared), error_output)
        
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
        elif term == "inner_product_s_restricted":
            self.operator["inner_product_s_restricted"] = self.inner_product["s"]
            return self.operator["inner_product_s_restricted"]
        else:
            return ParametrizedReducedDifferentialProblem.assemble_operator(self, term, current_stage)
                                
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

