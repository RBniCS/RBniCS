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

from __future__ import print_function
from rbnics.problems.base import ParametrizedReducedDifferentialProblem
from rbnics.problems.stokes_optimal_control.stokes_optimal_control_problem import StokesOptimalControlProblem
from rbnics.backends import LinearSolver, product, sum, transpose
from rbnics.backends.online import OnlineFunction
from rbnics.utils.decorators import Extends, override, ReducedProblemFor, MultiLevelReducedProblem
from rbnics.reduction_methods.stokes_optimal_control import StokesOptimalControlReductionMethod
from rbnics.utils.mpi import print

@Extends(ParametrizedReducedDifferentialProblem) # needs to be first in order to override for last the methods.
@ReducedProblemFor(StokesOptimalControlProblem, StokesOptimalControlReductionMethod)
@MultiLevelReducedProblem
class StokesOptimalControlReducedProblem(ParametrizedReducedDifferentialProblem):
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        ParametrizedReducedDifferentialProblem.__init__(self, truth_problem, **kwargs)
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        
    # Perform an online solve (internal)
    def _solve(self, N, **kwargs):
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
        
    # Perform an online evaluation of the cost functional
    @override
    def _compute_output(self, N):
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
                
        # Custom combination of inner products *not* to add inner product corresponding to supremizers
        def _combine_all_inner_products(self):
            # Temporarily change self.components
            components_bak = self.components
            self.components = ["v", "p", "u", "w", "q"]
            # Call Parent
            StokesOptimalControlReducedProblem_Base._combine_all_inner_products(self)
            # Restore
            self.components = components_bak
            
        # Custom combination of inner products *not* to add projection inner product corresponding to supremizers
        def _combine_all_projection_inner_products(self):
            # Temporarily change self.components
            components_bak = self.components
            self.components = ["v", "p", "u", "w", "q"]
            # Call Parent
            StokesOptimalControlReducedProblem_Base._combine_all_projection_inner_products(self)
            # Restore
            self.components = components_bak
