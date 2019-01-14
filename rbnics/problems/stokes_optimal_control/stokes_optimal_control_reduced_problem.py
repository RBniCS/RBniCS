# Copyright (C) 2015-2019 by the RBniCS authors
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

from rbnics.problems.base import LinearReducedProblem
from rbnics.backends import product, sum, transpose

def StokesOptimalControlReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    StokesOptimalControlReducedProblem_Base = LinearReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass)

    class StokesOptimalControlReducedProblem_Class(StokesOptimalControlReducedProblem_Base):
        
        class ProblemSolver(StokesOptimalControlReducedProblem_Base.ProblemSolver):
            def matrix_eval(self):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                for term in ("a", "a*", "b", "b*", "bt", "bt*", "c", "c*", "m", "n"):
                    assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term][:N, :N]))
                return (
                      assembled_operator["m"]                                                      + assembled_operator["a*"] + assembled_operator["bt*"]
                                                                                                   + assembled_operator["b*"]
                                                                         + assembled_operator["n"] - assembled_operator["c*"]
                    + assembled_operator["a"] + assembled_operator["bt"] - assembled_operator["c"]
                    + assembled_operator["b"]
                )
                
            def vector_eval(self):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                for term in ("f", "g", "l"):
                    assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term][:N]))
                return (
                      assembled_operator["g"]
                    
                    
                    + assembled_operator["f"]
                    + assembled_operator["l"]
                )
                
            # Custom combination of boundary conditions *not* to add BCs of supremizers
            def bc_eval(self):
                problem = self.problem
                # Temporarily change problem.components
                components_bak = problem.components
                problem.components = ["v", "p", "w", "q"]
                # Call Parent
                bcs = StokesOptimalControlReducedProblem_Base.ProblemSolver.bc_eval(self)
                # Restore and return
                problem.components = components_bak
                return bcs
            
        # Perform an online evaluation of the cost functional
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
                    raise ValueError("Invalid value for order of term " + term)
            self._output = (
                0.5*(transpose(self._solution)*assembled_operator["m"]*self._solution) +
                0.5*(transpose(self._solution)*assembled_operator["n"]*self._solution) -
                transpose(assembled_operator["g"])*self._solution +
                0.5*assembled_operator["h"]
            )
        
        # If a value of N was provided, make sure to double it when dealing with y and p, due to
        # the aggregated component approach
        def _online_size_from_kwargs(self, N, **kwargs):
            if N is None:
                # then either,
                # * the user has passed kwargs, so we trust that he/she has doubled velocities, supremizers and pressures for us
                # * or self.N was copied, which already stores the correct count of basis functions
                return StokesOptimalControlReducedProblem_Base._online_size_from_kwargs(self, N, **kwargs)
            else:
                # then the integer value provided to N would be used for all components: need to double
                # it for velocities, supremizers and pressures
                N, kwargs = StokesOptimalControlReducedProblem_Base._online_size_from_kwargs(self, N, **kwargs)
                for component in ("v", "s", "p", "w", "r", "q"):
                    N[component] *= 2
                return N, kwargs
        
        # Internal method for error computation
        def _compute_error(self, **kwargs):
            components = ["v", "p", "u", "w", "q"] # but not supremizers
            if "components" not in kwargs:
                kwargs["components"] = components
            else:
                assert kwargs["components"] == components
            return StokesOptimalControlReducedProblem_Base._compute_error(self, **kwargs)
            
        # Internal method for relative error computation
        def _compute_relative_error(self, absolute_error, **kwargs):
            components = ["v", "p", "u", "w", "q"] # but not supremizers
            if "components" not in kwargs:
                kwargs["components"] = components
            else:
                assert kwargs["components"] == components
            return StokesOptimalControlReducedProblem_Base._compute_relative_error(self, absolute_error, **kwargs)
            
        # Assemble the reduced order affine expansion
        def assemble_operator(self, term, current_stage="online"):
            if term == "bt_restricted":
                self.operator["bt_restricted"] = self.operator["bt"]
                return self.operator["bt_restricted"]
            elif term == "bt*_restricted":
                self.operator["bt*_restricted"] = self.operator["bt*"]
                return self.operator["bt*_restricted"]
            elif term == "inner_product_s":
                self.inner_product["s"] = self.inner_product["v"]
                return self.inner_product["s"]
            elif term == "inner_product_r":
                self.inner_product["r"] = self.inner_product["w"]
                return self.inner_product["r"]
            elif term == "projection_inner_product_s":
                self.projection_inner_product["s"] = self.projection_inner_product["v"]
                return self.projection_inner_product["s"]
            elif term == "projection_inner_product_r":
                self.projection_inner_product["r"] = self.projection_inner_product["w"]
                return self.projection_inner_product["r"]
            else:
                return StokesOptimalControlReducedProblem_Base.assemble_operator(self, term, current_stage)
                
        # Custom combination of inner products *not* to add inner product corresponding to supremizers
        def _combine_all_inner_products(self):
            # Temporarily change self.components
            components_bak = self.components
            self.components = ["v", "p", "u", "w", "q"]
            # Call Parent
            combined_inner_products = StokesOptimalControlReducedProblem_Base._combine_all_inner_products(self)
            # Restore and return
            self.components = components_bak
            return combined_inner_products
            
        # Custom combination of inner products *not* to add projection inner product corresponding to supremizers
        def _combine_all_projection_inner_products(self):
            # Temporarily change self.components
            components_bak = self.components
            self.components = ["v", "p", "u", "w", "q"]
            # Call Parent
            combined_projection_inner_products = StokesOptimalControlReducedProblem_Base._combine_all_projection_inner_products(self)
            # Restore and return
            self.components = components_bak
            return combined_projection_inner_products
        
    # return value (a class) for the decorator
    return StokesOptimalControlReducedProblem_Class
