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
from rbnics.backends import assign, copy, product, sum, transpose
from rbnics.backends.online import OnlineFunction, OnlineLinearSolver
from rbnics.utils.cache import Cache
from rbnics.utils.io import OnlineSizeDict

def StokesReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    StokesReducedProblem_Base = LinearReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass)

    # Base class containing the interface of a projection based ROM
    # for saddle point problems.
    class StokesReducedProblem_Class(StokesReducedProblem_Base):
        
        def __init__(self, truth_problem, **kwargs):
            StokesReducedProblem_Base.__init__(self, truth_problem, **kwargs)
            # Auxiliary storage for solution of reduced order supremizer problem (if requested through solve_supremizer)
            self._supremizer = None # OnlineFunction
            # I/O
            def _supremizer_cache_key_generator(*args, **kwargs):
                assert len(args) == 2
                assert args[0] == self.mu
                return self._supremizer_cache_key_from_N_and_kwargs(args[1], **kwargs)
            self._supremizer_cache = Cache(
                "reduced problems",
                key_generator=_supremizer_cache_key_generator
            )
            
        class ProblemSolver(StokesReducedProblem_Base.ProblemSolver):
            def matrix_eval(self):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                for term in ("a", "b", "bt"):
                    assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term][:N, :N]))
                return assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"]
                
            def vector_eval(self):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                for term in ("f", "g"):
                    assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term][:N]))
                return assembled_operator["f"] + assembled_operator["g"]
                
            # Custom combination of boundary conditions *not* to add BCs of supremizers
            def bc_eval(self):
                problem = self.problem
                # Temporarily change problem.components
                components_bak = problem.components
                problem.components = ["u", "p"]
                # Call Parent
                bcs = StokesReducedProblem_Base.ProblemSolver.bc_eval(self)
                # Restore and return
                problem.components = components_bak
                return bcs
                
        def solve_supremizer(self, solution):
            N_us = OnlineSizeDict(solution.N) # create a copy
            del N_us["p"]
            kwargs = self._latest_solve_kwargs
            self._supremizer = OnlineFunction(N_us)
            try:
                assign(self._supremizer, self._supremizer_cache[self.mu, N_us, kwargs]) # **kwargs is not supported by __getitem__
            except KeyError:
                self._solve_supremizer(solution)
                self._supremizer_cache[self.mu, N_us, kwargs] = copy(self._supremizer)
            return self._supremizer
            
        def _solve_supremizer(self, solution):
            N_us = self._supremizer.N
            N_usp = solution.N
            assert len(self.inner_product["s"]) == 1 # the affine expansion storage contains only the inner product matrix
            assembled_operator_lhs = self.inner_product["s"][0][:N_us, :N_us]
            assembled_operator_bt = sum(product(self.compute_theta("bt_restricted"), self.operator["bt_restricted"][:N_us, :N_usp]))
            assembled_operator_rhs = assembled_operator_bt*solution
            if self.dirichlet_bc["u"] and not self.dirichlet_bc_are_homogeneous["u"]:
                assembled_dirichlet_bc = dict()
                assert self.dirichlet_bc["s"]
                assert self.dirichlet_bc_are_homogeneous["s"]
                assembled_dirichlet_bc["u"] = self.compute_theta("dirichlet_bc_s")
            else:
                assembled_dirichlet_bc = None
            solver = OnlineLinearSolver(
                assembled_operator_lhs,
                self._supremizer,
                assembled_operator_rhs,
                assembled_dirichlet_bc
            )
            solver.set_parameters(self._linear_solver_parameters)
            solver.solve()
            
        def _supremizer_cache_key_from_N_and_kwargs(self, N, **kwargs):
            return self._cache_key_from_N_and_kwargs(N, **kwargs)
            
        # Internal method for error computation
        def _compute_error(self, **kwargs):
            components = ["u", "p"] # but not "s"
            if "components" not in kwargs:
                kwargs["components"] = components
            else:
                assert kwargs["components"] == components
            return StokesReducedProblem_Base._compute_error(self, **kwargs)
            
        # Internal method for relative error computation
        def _compute_relative_error(self, absolute_error, **kwargs):
            components = ["u", "p"] # but not "s"
            if "components" not in kwargs:
                kwargs["components"] = components
            else:
                assert kwargs["components"] == components
            return StokesReducedProblem_Base._compute_relative_error(self, absolute_error, **kwargs)
            
        def export_supremizer(self, folder=None, filename=None, supremizer=None, component=None, suffix=None):
            if supremizer is None:
                supremizer = self._supremizer
            N_us = supremizer.N
            basis_functions_us = self.basis_functions[["u", "s"]]
            self.truth_problem.export_supremizer(folder, filename, basis_functions_us[:N_us]*supremizer, component, suffix)
            
        # Assemble the reduced order affine expansion
        def assemble_operator(self, term, current_stage="online"):
            if current_stage == "offline":
                if term == "bt_restricted":
                    basis_functions_us = self.basis_functions[["u", "s"]]
                    assert self.Q["bt_restricted"] == self.truth_problem.Q["bt_restricted"]
                    for q in range(self.Q["bt_restricted"]):
                        self.operator["bt_restricted"][q] = transpose(basis_functions_us)*self.truth_problem.operator["bt_restricted"][q]*self.basis_functions
                    self.operator["bt_restricted"].save(self.folder["reduced_operators"], "operator_bt_restricted")
                    return self.operator["bt_restricted"]
                elif term == "inner_product_s":
                    basis_functions_us = self.basis_functions[["u", "s"]]
                    assert len(self.inner_product["s"]) == 1 # the affine expansion storage contains only the inner product matrix
                    assert len(self.truth_problem.inner_product["s"]) == 1 # the affine expansion storage contains only the inner product matrix
                    self.inner_product["s"][0] = transpose(basis_functions_us)*self.truth_problem.inner_product["s"][0]*basis_functions_us
                    self.inner_product["s"].save(self.folder["reduced_operators"], "inner_product_s")
                    return self.inner_product["s"]
                elif term == "projection_inner_product_s":
                    basis_functions_us = self.basis_functions[["u", "s"]]
                    assert len(self.projection_inner_product["s"]) == 1 # the affine expansion storage contains only the inner product matrix
                    assert len(self.truth_problem.projection_inner_product["s"]) == 1 # the affine expansion storage contains only the inner product matrix
                    self.projection_inner_product["s"][0] = transpose(basis_functions_us)*self.truth_problem.projection_inner_product["s"][0]*basis_functions_us
                    self.projection_inner_product["s"].save(self.folder["reduced_operators"], "projection_inner_product_s")
                    return self.projection_inner_product["s"]
                else:
                    return StokesReducedProblem_Base.assemble_operator(self, term, current_stage)
            else:
                return StokesReducedProblem_Base.assemble_operator(self, term, current_stage)
                
        # Custom combination of inner products *not* to add inner product corresponding to supremizers
        def _combine_all_inner_products(self):
            # Temporarily change self.components
            components_bak = self.components
            self.components = ["u", "p"]
            # Call Parent
            combined_inner_products = StokesReducedProblem_Base._combine_all_inner_products(self)
            # Restore and return
            self.components = components_bak
            return combined_inner_products
            
        # Custom combination of inner products *not* to add projection inner product corresponding to supremizers
        def _combine_all_projection_inner_products(self):
            # Temporarily change self.components
            components_bak = self.components
            self.components = ["u", "p"]
            # Call Parent
            combined_projection_inner_products = StokesReducedProblem_Base._combine_all_projection_inner_products(self)
            # Restore and return
            self.components = components_bak
            return combined_projection_inner_products
        
    # return value (a class) for the decorator
    return StokesReducedProblem_Class
