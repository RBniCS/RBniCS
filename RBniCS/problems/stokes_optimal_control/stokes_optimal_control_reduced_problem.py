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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from RBniCS.problems.base import ParametrizedReducedDifferentialProblem
from RBniCS.problems.stokes_optimal_control.stokes_optimal_control_problem import StokesOptimalControlProblem
from RBniCS.backends import LinearSolver, product, sum, transpose
from RBniCS.backends.online import OnlineFunction
from RBniCS.utils.decorators import Extends, override, ReducedProblemFor, MultiLevelReducedProblem
from RBniCS.reduction_methods.stokes_optimal_control import StokesOptimalControlReductionMethod
from RBniCS.utils.mpi import print

@Extends(ParametrizedReducedDifferentialProblem) # needs to be first in order to override for last the methods.
@ReducedProblemFor(StokesOptimalControlProblem, StokesOptimalControlReductionMethod)
@MultiLevelReducedProblem
class StokesOptimalControlReducedProblem(ParametrizedReducedDifferentialProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        ParametrizedReducedDifferentialProblem.__init__(self, truth_problem, **kwargs)
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # I/O
        self.folder["state_supremizer_snapshots"] = self.folder_prefix + "/" + "snapshots"
        self.folder["adjoint_supremizer_snapshots"] = self.folder_prefix + "/" + "snapshots"
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
        
    # Perform an online solve (internal)
    def _solve(self, N, **kwargs):
        N += self.N_bc
        assembled_operator = dict()
        for term in ("a", "a*", "b", "b*", "bt", "bt*", "c", "c*", "m", "n", "f", "g", "l"):
            assert self.terms_order[term] in (1, 2)
            if self.terms_order[term] == 2:
                assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term][:N, :N]))
            elif self.terms_order[term] == 1:
                assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term][:N]))
            else:
                raise AssertionError("Invalid value for order of term " + term)
        theta_bc = dict()
        for component in ("v", "p", "w", "q"):
            if self.dirichlet_bc[component] and not self.dirichlet_bc_are_homogeneous[component]:
                theta_bc[component] = self.compute_theta("dirichlet_bc_" + component)
        assert self.dirichlet_bc["u"] is False, "Control should not be constrained by Dirichlet BCs"
        if len(theta_bc) == 0:
            theta_bc = None
        self._solution = OnlineFunction(N)
        solver = LinearSolver(
            (
                  assembled_operator["m"]                                                      + assembled_operator["a*"] + assembled_operator["bt*"]
                                                                                               + assembled_operator["b*"]
                                                                     + assembled_operator["n"] - assembled_operator["c*"]
                + assembled_operator["a"] + assembled_operator["bt"] - assembled_operator["c"]
                + assembled_operator["b"]
            ),
            self._solution,
            (
                  assembled_operator["g"]
                
                
                + assembled_operator["f"]
                + assembled_operator["l"]
            ),
            theta_bc
        )
        solver.solve()
        return self._solution
        
    # Perform an online evaluation of the cost functional
    @override
    def output(self):
        N = self._solution.N
        assembled_operator = dict()
        for term in ("m", "n", "g", "h"):
            assert self.terms_order[term] in (0, 1, 2)
            if self.terms_order[term] == 2:
                assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term][:N, :N]))
            elif self.terms_order[term] == 1:
                assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term][:N]))
            elif self.terms_order[term] == 0:
                assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term]))
            else:
                raise AssertionError("Invalid value for order of term " + term)
        self._output = (
            0.5*(transpose(self._solution)*assembled_operator["m"]*self._solution) + 
            0.5*(transpose(self._solution)*assembled_operator["n"]*self._solution) - 
            transpose(assembled_operator["g"])*self._solution + 
            0.5*assembled_operator["h"]
        )
        return self._output
    
    # If a value of N was provided, make sure to double it when dealing with y and p, due to
    # the aggregated component approach
    @override
    def _online_size_from_kwargs(self, N, **kwargs):
        all_components_in_kwargs = all([c in kwargs for c in self.components])
        if N is None:
            # then either,
            # * the user has passed kwargs, so we trust that he/she has doubled velocities, supremizers and pressures for us
            # * or self.N was copied, which already stores the correct count of basis functions
            return ParametrizedReducedDifferentialProblem._online_size_from_kwargs(self, N, **kwargs)
        else:
            # then the integer value provided to N would be used for all components: need to double
            # it for velocities, supremizers and pressures
            N, kwargs = ParametrizedReducedDifferentialProblem._online_size_from_kwargs(self, N, **kwargs)
            for component in ("v", "s", "p", "w", "r", "q"):
                N[component] *= 2
            return N, kwargs
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Internal method for error computation
    @override
    def _compute_error(self, **kwargs):
        components = ["v", "p", "u", "w", "q"] # but not supremizers
        if "components" not in kwargs:
            kwargs["components"] = components
        else:
            assert kwargs["components"] == components
        return ParametrizedReducedDifferentialProblem._compute_error(self, **kwargs)
        
    # Internal method for relative error computation
    @override
    def _compute_relative_error(self, absolute_error, **kwargs):
        components = ["v", "p", "u", "w", "q"] # but not supremizers
        if "components" not in kwargs:
            kwargs["components"] = components
        else:
            assert kwargs["components"] == components
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
        elif term == "bt*_restricted":
            self.operator["bt*_restricted"] = self.operator["bt*"]
            return self.operator["bt_restricted"]
        elif term == "inner_product_s":
            self.inner_product["s"] = self.inner_product["v"]
            return self.inner_product["s"]
        elif term == "inner_product_r":
            self.inner_product["r"] = self.inner_product["w"]
            return self.inner_product["r"]
        else:
            return ParametrizedReducedDifferentialProblem.assemble_operator(self, term, current_stage)
                                
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
    ## Postprocess a snapshot before adding it to the basis/snapshot matrix: also solve the supremizer problems
    def postprocess_snapshot(self, snapshot, snapshot_index):
        # Compute supremizers
        print("state supremizer solve for mu =", self.truth_problem.mu)
        state_supremizer = self.truth_problem.solve_state_supremizer()
        self.truth_problem.export_solution(self.folder["state_supremizer_snapshots"], "truth_" + str(snapshot_index) + "_s", state_supremizer, component="s")
        print("adjoint supremizer solve for mu =", self.truth_problem.mu)
        adjoint_supremizer = self.truth_problem.solve_adjoint_supremizer()
        self.truth_problem.export_solution(self.folder["adjoint_supremizer_snapshots"], "truth_" + str(snapshot_index) + "_r", adjoint_supremizer, component="r")
        # Call parent
        snapshot = ParametrizedReducedDifferentialProblem.postprocess_snapshot(self, snapshot, snapshot_index)
        # Return a tuple
        return (snapshot, state_supremizer, adjoint_supremizer)
        
