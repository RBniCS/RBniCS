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

from abc import ABCMeta, abstractmethod
from RBniCS.backends import AffineExpansionStorage, BasisFunctionsMatrix, Function, FunctionsList, LinearSolver, product, sum, transpose
from RBniCS.backends.online import OnlineAffineExpansionStorage
from RBniCS.utils.decorators import Extends, override

def RBReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class RBReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        __metaclass__ = ABCMeta
        
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the reduced order model object
        #  @{
        
        ## Default initialization of members.
        @override
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
            # I/O
            self.folder["error_estimation"] = self.folder_prefix + "/" + "error_estimation"
            
            # Provide a default value for Riesz terms and Riesz product terms
            self.riesz_terms = [term for term in self.terms]
            self.riesz_product_terms = [(term1, term2) for term1 in self.terms for term2 in self.terms if self.terms_order[term1] >= self.terms_order[term2]]
            
        #  @}
        ########################### end - CONSTRUCTORS - end ########################### 
        
        ###########################     ONLINE STAGE     ########################### 
        ## @defgroup OnlineStage Methods related to the online stage
        #  @{
        
        ## Initialize data structures required for the online phase
        @override
        def init(self, current_stage="online"):
            ParametrizedReducedDifferentialProblem_DerivedClass.init(self, current_stage)
            self._init_error_estimation_operators(current_stage)
            
        def _init_error_estimation_operators(self, current_stage="online"):
            # Also initialize data structures related to error estimation
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
                self._init_riesz_solve_inner_product()
                # Also setup homogeneous Dirichlet BCs, if any
                self._init_riesz_solve_homogeneous_dirichlet_bc()
            else:
                raise AssertionError("Invalid stage in _init_error_estimation_operators().")
                
        def _init_riesz_solve_inner_product(self):
            if len(self.components) > 1:
                all_inner_products = list()
                for component in self.components:
                    assert len(self.truth_problem.inner_product[component]) == 1 # the affine expansion storage contains only the inner product matrix
                    all_inner_products.append(self.truth_problem.inner_product[component][0])
                all_inner_products = tuple(all_inner_products)
            else:
                assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                all_inner_products = (self.truth_problem.inner_product[0], )
            all_inner_products = AffineExpansionStorage(all_inner_products)
            all_inner_products_thetas = (1.,)*len(all_inner_products)
            self._riesz_solve_inner_product = sum(product(all_inner_products_thetas, all_inner_products))
        
        def _init_riesz_solve_homogeneous_dirichlet_bc(self):
            if len(self.components) > 1:
                all_truth_dirichlet_bcs = list()
                for component in self.components:
                    if self.truth_problem.dirichlet_bc[component] is not None:
                        all_truth_dirichlet_bcs.extend(self.truth_problem.dirichlet_bc[component])
                if len(all_truth_dirichlet_bcs) > 0:
                    all_truth_dirichlet_bcs = tuple(all_truth_dirichlet_bcs)
                    all_truth_dirichlet_bcs = AffineExpansionStorage(all_truth_dirichlet_bcs)
                else:
                    all_truth_dirichlet_bcs = None
            else:
                all_truth_dirichlet_bcs = self.truth_problem.dirichlet_bc
            if all_truth_dirichlet_bcs is not None:
                all_truth_dirichlet_bcs_thetas = (0.,)*len(all_truth_dirichlet_bcs)
                self._riesz_solve_homogeneous_dirichlet_bc = sum(product(all_truth_dirichlet_bcs_thetas, all_truth_dirichlet_bcs))
            else:
                self._riesz_solve_homogeneous_dirichlet_bc = None
        
        ## Return an error bound for the current solution
        @abstractmethod
        def estimate_error(self):
            raise NotImplementedError("The method estimate_error() is problem-specific and needs to be overridden.")
            
        ## Return a relative error bound for the current solution
        @abstractmethod
        def estimate_relative_error(self):
            raise NotImplementedError("The method estimate_relative_error() is problem-specific and needs to be overridden.")
        
        ## Return an error bound for the current output. Provides a default implementation which is consistent with the default
        ## output computation.
        def estimate_error_output(self):
            return NotImplemented
            
        ## Return an error bound for the current output. Provides a default implementation which is consistent with the default
        ## output computation.
        def estimate_relative_error_output(self):
            return NotImplemented
            
        #  @}
        ########################### end - ONLINE STAGE - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
        
        ## Build operators for error estimation
        def build_error_estimation_operators(self):
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
                
        ## Compute the Riesz representation of term
        def compute_riesz(self, term):
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
                                self.riesz[term][q].enrich(self._riesz_solve_storage, component=component)
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
                raise AssertionError("Invalid value for order of term " + term)
            
        #  @}
        ########################### end - OFFLINE STAGE - end ########################### 
        
        ###########################     PROBLEM SPECIFIC     ########################### 
        ## @defgroup ProblemSpecific Problem specific methods
        #  @{
        
        ## Assemble operators for error estimation
        def assemble_error_estimation_operators(self, term, current_stage="online"):
            assert current_stage in ("online", "offline")
            assert isinstance(term, tuple)
            assert len(term) == 2
            # 
            if current_stage == "online": # load from file
                if not term in self.riesz_product:
                    self.riesz_product[term] = OnlineAffineExpansionStorage(0, 0) # it will be resized by load
                self.riesz_product[term].load(self.folder["error_estimation"], "riesz_product_" + term[0] + "_" + term[1])
                return self.riesz_product[term]
            elif current_stage == "offline":
                assert self.terms_order[term[0]] in (1, 2)
                assert self.terms_order[term[1]] in (1, 2)
                assert self.terms_order[term[0]] >= self.terms_order[term[1]], "Please swap the order of " + str(term) + " in self.riesz_product_terms" # otherwise for (term1, term2) of orders (1, 2) we would have a row vector, rather than a column one
                if self.terms_order[term[0]] == 2 and self.terms_order[term[1]] == 2:
                    for q0 in range(self.Q[term[0]]):
                        for q1 in range(self.Q[term[1]]):
                            self.riesz_product[term][q0, q1] = transpose(self.riesz[term[0]][q0])*self._riesz_solve_inner_product*self.riesz[term[1]][q1]
                elif self.terms_order[term[0]] == 2 and self.terms_order[term[1]] == 1:
                    for q0 in range(self.Q[term[0]]):
                        for q1 in range(self.Q[term[1]]):
                            assert len(self.riesz[term[1]][q1]) == 1
                            self.riesz_product[term][q0, q1] = transpose(self.riesz[term[0]][q0])*self._riesz_solve_inner_product*self.riesz[term[1]][q1][0]
                elif self.terms_order[term[0]] == 1 and self.terms_order[term[1]] == 1:
                    for q0 in range(self.Q[term[0]]):
                        assert len(self.riesz[term[0]][q0]) == 1
                        for q1 in range(self.Q[term[1]]):
                            assert len(self.riesz[term[1]][q1]) == 1
                            self.riesz_product[term][q0, q1] = transpose(self.riesz[term[0]][q0][0])*self._riesz_solve_inner_product*self.riesz[term[1]][q1][0]
                else:
                    raise ValueError("Invalid term order for assemble_error_estimation_operators().")
                self.riesz_product[term].save(self.folder["error_estimation"], "riesz_product_" + term[0] + "_" + term[1])
                return self.riesz_product[term]
            else:
                raise AssertionError("Invalid stage in assemble_error_estimation_operators().")
                
        #  @}
        ########################### end - PROBLEM SPECIFIC - end ########################### 
        
    # return value (a class) for the decorator
    return RBReducedProblem_Class
    
