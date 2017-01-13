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
    
    # Internal method for error computation
    @override
    def _compute_error(self):
        components = ["u", "p"] # but not "s"
        assert "components" not in kwargs
        kwargs["components"] = components
        return ParametrizedReducedDifferentialProblem._compute_error(self, **kwargs)
        
    # Internal method for relative error computation
    @override
    def _compute_relative_error(self, absolute_error, **kwargs):
        components = ["u", "p"] # but not "s"
        assert "components" not in kwargs
        kwargs["components"] = components
        return ParametrizedReducedDifferentialProblem._compute_relative_error(self, absolute_error, **kwargs)
        
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

