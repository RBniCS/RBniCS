# Copyright (C) 2015-2018 by the RBniCS authors
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

from abc import ABCMeta, abstractmethod
import os
from rbnics.backends import BasisFunctionsMatrix, Function, FunctionsList, LinearSolver, transpose
from rbnics.backends.online import OnlineAffineExpansionStorage
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators

@RequiredBaseDecorators(None)
def RBReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    @PreserveClassName
    class RBReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass, metaclass=ABCMeta):
        """
        Abstract class. All the terms for error estimator are initialized.
        
        :param truth_problem: class of the truth problem to be solved.
        """
        
        # Default initialization of members.
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # $$ ONLINE DATA STRUCTURES $$ #
            # Residual terms
            self.riesz = dict() # from string to FunctionsList
            self.riesz_terms = list()
            self.riesz_product = dict() # from string to OnlineAffineExpansionStorage
            self.riesz_product_terms = list() # of tuple
            self.build_error_estimation_operators__initialized = False
            
            # $$ OFFLINE DATA STRUCTURES $$ #
            # Residual terms
            self._riesz_solve_storage = Function(self.truth_problem.V)
            self._riesz_solve_inner_product = None # setup by init()
            self._riesz_solve_homogeneous_dirichlet_bc = None # setup by init()
            self._riesz_product_inner_product = None # setup by init()
            # I/O
            self.folder["error_estimation"] = os.path.join(self.folder_prefix, "error_estimation")
            
            # Provide a default value for Riesz terms and Riesz product terms
            self.riesz_terms = [term for term in self.terms]
            self.riesz_product_terms = [(term1, term2) for term1 in self.terms for term2 in self.terms if self.terms_order[term1] >= self.terms_order[term2]]
        
        def init(self, current_stage="online"):
            """
            Initialize data structures required for the online phase.
            """
            ParametrizedReducedDifferentialProblem_DerivedClass.init(self, current_stage)
            self._init_error_estimation_operators(current_stage)
            
        def _init_error_estimation_operators(self, current_stage="online"):
            """
            Initialize data structures related to error estimation.
            """
            assert current_stage in ("online", "offline")
            if current_stage == "online":
                for term in self.riesz_product_terms:
                    self.riesz_product[term] = self.assemble_error_estimation_operators(term, "online")
            elif current_stage == "offline":
                for term in self.riesz_terms:
                    self.riesz[term] = OnlineAffineExpansionStorage(self.Q[term])
                    for q in range(self.Q[term]):
                        assert self.terms_order[term] in (1, 2)
                        if self.terms_order[term] > 1:
                            riesz_term_q = BasisFunctionsMatrix(self.truth_problem.V)
                            riesz_term_q.init(self.components)
                        else:
                            riesz_term_q = FunctionsList(self.truth_problem.V) # will be of size 1
                        self.riesz[term][q] = riesz_term_q
                for term in self.riesz_product_terms:
                    self.riesz_product[term] = OnlineAffineExpansionStorage(self.Q[term[0]], self.Q[term[1]])
                # Also initialize inner product for Riesz solve
                self._riesz_solve_inner_product = self.truth_problem._combined_inner_product
                # Also setup homogeneous Dirichlet BCs for Riesz solve, if any
                self._riesz_solve_homogeneous_dirichlet_bc = self.truth_problem._combined_and_homogenized_dirichlet_bc
                # Also initialize inner product for Riesz products. This is the same as the inner product for Riesz solves
                # but setting to zero rows & columns associated to boundary conditions
                if self._riesz_solve_homogeneous_dirichlet_bc is not None:
                    self._riesz_product_inner_product = self._riesz_solve_inner_product & ~self._riesz_solve_homogeneous_dirichlet_bc
                else:
                    self._riesz_product_inner_product = self._riesz_solve_inner_product
            else:
                raise ValueError("Invalid stage in _init_error_estimation_operators().")
                
        @abstractmethod
        def estimate_error(self):
            """
            Returns an error bound for the current solution.
            """
            raise NotImplementedError("The method estimate_error() is problem-specific and needs to be overridden.")
            
        @abstractmethod
        def estimate_relative_error(self):
            """
            It returns a relative error bound for the current solution.
            """
            raise NotImplementedError("The method estimate_relative_error() is problem-specific and needs to be overridden.")
        
        def estimate_error_output(self):
            """
            It returns an error bound for the current output.
            """
            return NotImplemented
            
        def estimate_relative_error_output(self):
            """
            It returns an relative error bound for the current output.
            """
            return NotImplemented
   
        def build_error_estimation_operators(self):
            """
            It builds operators for error estimation.
            """
            if not self.build_error_estimation_operators__initialized: # this part does not depend on N, so we compute it only once
                for term in self.riesz_terms:
                    if self.terms_order[term] == 1:
                        # Compute the Riesz representation of terms that do not depend on the solution
                        self.compute_riesz(term)
                        # Compute the (term, term) Riesz representors product
                        self.assemble_error_estimation_operators((term, term), "offline")
                        #
                        self.build_error_estimation_operators__initialized = True
            
            # Update the Riesz representation with the new basis function(s)
            for term in self.riesz_terms:
                if self.terms_order[term] > 1:
                    self.compute_riesz(term)
            
            # Update the (term1, term2) Riesz representors product with the new basis function
            for term in self.riesz_product_terms:
                self.assemble_error_estimation_operators(term, "offline")
        
        def compute_riesz(self, term):
            """
            It computes the Riesz representation of term.
            
            :param term: the forms of the truth problem.
            """
            # Compute the Riesz representor
            assert self.terms_order[term] in (1, 2)
            if self.terms_order[term] == 1:
                for q in range(self.Q[term]):
                    solver = LinearSolver(
                        self._riesz_solve_inner_product,
                        self._riesz_solve_storage,
                        self.truth_problem.operator[term][q],
                        self._riesz_solve_homogeneous_dirichlet_bc
                    )
                    solver.solve()
                    self.riesz[term][q].enrich(self._riesz_solve_storage)
            elif self.terms_order[term] == 2:
                for q in range(self.Q[term]):
                    if len(self.components) > 1:
                        for component in self.components:
                            for n in range(len(self.riesz[term][q][component]), self.N[component] + self.N_bc[component]):
                                solver = LinearSolver(
                                    self._riesz_solve_inner_product,
                                    self._riesz_solve_storage,
                                    -1.*self.truth_problem.operator[term][q]*self.Z[component][n],
                                    self._riesz_solve_homogeneous_dirichlet_bc
                                )
                                solver.solve()
                                self.riesz[term][q].enrich(self._riesz_solve_storage, component={None: component})
                    else:
                        for n in range(len(self.riesz[term][q]), self.N + self.N_bc):
                            solver = LinearSolver(
                                self._riesz_solve_inner_product,
                                self._riesz_solve_storage,
                                -1.*self.truth_problem.operator[term][q]*self.Z[n],
                                self._riesz_solve_homogeneous_dirichlet_bc
                            )
                            solver.solve()
                            self.riesz[term][q].enrich(self._riesz_solve_storage)
            else:
                raise ValueError("Invalid value for order of term " + term)
        
        def assemble_error_estimation_operators(self, term, current_stage="online"):
            """
            It assembles operators for error estimation.
            """
            assert current_stage in ("online", "offline")
            assert isinstance(term, tuple)
            assert len(term) == 2
            if current_stage == "online": # load from file
                if term not in self.riesz_product:
                    self.riesz_product[term] = OnlineAffineExpansionStorage(0, 0) # it will be resized by load
                assert "error_estimation" in self.folder
                self.riesz_product[term].load(self.folder["error_estimation"], "riesz_product_" + term[0] + "_" + term[1])
                return self.riesz_product[term]
            elif current_stage == "offline":
                assert self.terms_order[term[0]] in (1, 2)
                assert self.terms_order[term[1]] in (1, 2)
                assert self.terms_order[term[0]] >= self.terms_order[term[1]], "Please swap the order of " + str(term) + " in self.riesz_product_terms" # otherwise for (term1, term2) of orders (1, 2) we would have a row vector, rather than a column one
                if self.terms_order[term[0]] == 2 and self.terms_order[term[1]] == 2:
                    for q0 in range(self.Q[term[0]]):
                        for q1 in range(self.Q[term[1]]):
                            self.riesz_product[term][q0, q1] = transpose(self.riesz[term[0]][q0])*self._riesz_product_inner_product*self.riesz[term[1]][q1]
                elif self.terms_order[term[0]] == 2 and self.terms_order[term[1]] == 1:
                    for q0 in range(self.Q[term[0]]):
                        for q1 in range(self.Q[term[1]]):
                            assert len(self.riesz[term[1]][q1]) == 1
                            self.riesz_product[term][q0, q1] = transpose(self.riesz[term[0]][q0])*self._riesz_product_inner_product*self.riesz[term[1]][q1][0]
                elif self.terms_order[term[0]] == 1 and self.terms_order[term[1]] == 1:
                    for q0 in range(self.Q[term[0]]):
                        assert len(self.riesz[term[0]][q0]) == 1
                        for q1 in range(self.Q[term[1]]):
                            assert len(self.riesz[term[1]][q1]) == 1
                            self.riesz_product[term][q0, q1] = transpose(self.riesz[term[0]][q0][0])*self._riesz_product_inner_product*self.riesz[term[1]][q1][0]
                else:
                    raise ValueError("Invalid term order for assemble_error_estimation_operators().")
                if "error_estimation" in self.folder:
                    self.riesz_product[term].save(self.folder["error_estimation"], "riesz_product_" + term[0] + "_" + term[1])
                return self.riesz_product[term]
            else:
                raise ValueError("Invalid stage in assemble_error_estimation_operators().")
        
    # return value (a class) for the decorator
    return RBReducedProblem_Class
