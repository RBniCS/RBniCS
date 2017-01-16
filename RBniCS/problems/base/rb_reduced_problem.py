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
from RBniCS.backends import Function, FunctionsList
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
            self.riesz_product = dict() # from string to OnlineAffineExpansionStorage
            self.build_error_estimation_operators__initialized = False
            
            # $$ OFFLINE DATA STRUCTURES $$ #
            # Residual terms
            self._riesz_solve_storage = Function(self.truth_problem.V)
            # I/O
            self.folder["error_estimation"] = self.folder_prefix + "/" + "error_estimation"

            
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
                for (index1, term1) in enumerate(self.terms):
                    for (index2, term2) in enumerate(self.terms[index1:], start=index1):
                        self.riesz_product[term1 + term2] = self.assemble_error_estimation_operators("riesz_product_" + term1 + term2, "online")
            elif current_stage == "offline":
                for term in self.terms:
                    self.riesz[term] = OnlineAffineExpansionStorage(self.Q[term])
                    for q in range(self.Q[term]):
                        self.riesz[term][q] = FunctionsList(self.truth_problem.V)
                for (index1, term1) in enumerate(self.terms):
                    for (index2, term2) in enumerate(self.terms[index1:], start=index1):
                        self.riesz_product[term1 + term2] = OnlineAffineExpansionStorage(self.Q[term1], self.Q[term2])
            else:
                raise AssertionError("Invalid stage in _init_error_estimation_operators().")
        
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
                for term in self.terms:
                    if self.terms_order[term] == 1:
                        # Compute the Riesz representation of terms that do not depend on the solution
                        self.compute_riesz(term)
                        # Compute the (term, term) Riesz representors product
                        self.assemble_error_estimation_operators("riesz_product_" + term + term, "offline")
                        #
                        self.build_error_estimation_operators__initialized = True
            
            # Update the Riesz representation with the new basis function(s)
            for term in self.terms:
                if self.terms_order[term] > 1:
                    self.compute_riesz(term)
            
            # Update the (term1, term2) Riesz representors product with the new basis function
            for (index1, term1) in enumerate(self.terms):
                for (index2, term2) in enumerate(self.terms[index1:], start=index1):
                    if self.terms_order[term1] > 1 or self.terms_order[term2] > 1:
                        self.assemble_error_estimation_operators("riesz_product_" + term1 + term2, "offline")
                
        ## Compute the Riesz representation of term
        @abstractmethod
        def compute_riesz(self, term):
            raise NotImplementedError("The method compute_riesz() is problem-specific and needs to be overridden.")
                
        #  @}
        ########################### end - OFFLINE STAGE - end ########################### 
        
        ###########################     PROBLEM SPECIFIC     ########################### 
        ## @defgroup ProblemSpecific Problem specific methods
        #  @{
        
        ## Assemble operators for error estimation
        @abstractmethod
        def assemble_error_estimation_operators(self, term, current_stage="online"):
            raise NotImplementedError("The method assemble_error_estimation_operators() is problem-specific and needs to be overridden.")
                
        #  @}
        ########################### end - PROBLEM SPECIFIC - end ########################### 
        
    # return value (a class) for the decorator
    return RBReducedProblem_Class
    
