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
## @file eim.py
#  @brief Implementation of the empirical interpolation method for the interpolation of parametrized functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
import types
from RBniCS.problems import ParametrizedProblem
from RBniCS.linear_algebra import AffineExpansionOnlineStorage, FunctionsList

def ExactParametrizedFunctionEvaluationDecoratedReducedProblem(ReducedParametrizedProblem_DerivedClass):
    
    def _AlsoDecorateErrorEstimationOperators(ReducedParametrizedProblem_DecoratedClass):
        if hasattr(ReducedParametrizedProblem_DecoratedClass, "assemble_error_estimation_operators"):
            class _AlsoDecorateErrorEstimationOperators_Class(ReducedParametrizedProblem_DecoratedClass):
                def __init__(self, truth_problem):
                    # Call the parent initialization
                    ReducedParametrizedProblem_DecoratedClass.__init__(self, truth_problem)
                    # Avoid useless assemblies
                    self.get_eps2.__func__.previous_mu = None
                    self.get_eps2.__func__.previous_self_N = None
                    
                ###########################     ONLINE STAGE     ########################### 
                ## @defgroup OnlineStage Methods related to the online stage
                #  @{
            
                ## Initialize data structures required for the online phase
                def init(self, current_stage="online"):
                    if current_stage == "online":
                        # The offline/online separation does not hold anymore, so in 
                        # assemble_error_estimation_operators() we need to re-assemble operators. 
                        # Their data structure is initialized (with suitable size) in this 
                        # initialization method.
                        for term in self.truth_problem.operator:
                            self.riesz[term] = AffineExpansionOnlineStorage(self.truth_problem.Q[term])
                            for q in range(self.truth_problem.Q[term]):
                                self.riesz[term][q] = FunctionsList()
                        for term1 in self.truth_problem.operator:
                            for term2 in self.truth_problem.operator:
                                if term1 > term2: # alphabetical order
                                    continue
                                
                                self.riesz_product[term1 + term2] = AffineExpansionOnlineStorage(self.truth_problem.Q[term1], self.truth_problem.Q[term2])
                                # Make sure to disable the save() method of the operator, which is 
                                # called internally by assemble_operator() since it is not possible
                                # to precompute operators, and thus they should not be saved
                                def disabled_save(self, folder, filename):
                                    pass
                                self.riesz_product[term1 + term2].save = types.MethodType(disabled_save, self.riesz_product[term1 + term2])
                                # Make sure to raise an error if the load() method of the operator,
                                # since we have not saved anything and it should never be called
                                def error_load(self, folder, filename):
                                    raise AttributeError
                                self.riesz_product[term1 + term2].load = types.MethodType(error_load, self.riesz_product[term1 + term2])
                        # Then, call parent method, which in turn will call assemble_operator()
                        # Note that some of the assembled operators will be overwritten at
                        # each get_eps2() call.
                        self.current_stage = current_stage # would have been updated anyway when calling parent
                        self.build_error_estimation_operators() # as a workardound for asserts. It will be discarded
                        ReducedParametrizedProblem_DerivedClass.init(self, current_stage)
                    else:
                        # Call parent method
                        ReducedParametrizedProblem_DerivedClass.init(self, current_stage)
                        
                ## Return the numerator of the error bound for the current solution
                def get_eps2(self):
                    # The offline/online separation does not hold anymore, so, similarly to what we did in
                    # the truth problem, also at the reduced-order level we need to re-assemble operators,
                    # because the assemble_operator() *may* return parameter dependent operators.
                    assert(self._solve.__func__.previous_mu == self.mu) # get_eps2 is always called after _solve
                    if self.get_eps2.__func__.previous_mu != self.mu or self.get_eps2.__func__.previous_self_N != self.N:
                        current_stage_is_offline = (self.current_stage == "offline")
                        if current_stage_is_offline:
                            self.current_stage = "online" # temporary change
                        self.build_error_estimation_operators()
                        if current_stage_is_offline:
                            self.current_stage = "offline" # revert temporary change
                        # Avoid useless assemblies
                        self.get_eps2.__func__.previous_mu = self.mu
                        self.get_eps2.__func__.previous_self_N = self.N
                    return ReducedParametrizedProblem_DerivedClass.get_eps2(self)
                        
                #  @}
                ########################### end - ONLINE STAGE - end ########################### 
                
                ###########################     OFFLINE STAGE     ########################### 
                ## @defgroup OfflineStage Methods related to the offline stage
                #  @{
                    
                ## Build operators for error estimation
                def build_error_estimation_operators(self):
                    if self.current_stage == "online":
                        print("build operators for error estimation (due to inefficient evaluation)")
                        self.current_stage = "offline" # temporary change
                        for term in self.truth_problem.operator:
                            for q in range(self.truth_problem.Q[term]):
                                self.riesz[term][q].clear()
                        self.build_error_estimation_operators.__func__.initialized = False
                        ReducedParametrizedProblem_DecoratedClass.build_error_estimation_operators(self)
                        self.current_stage = "online" # revert temporary change
                    else:
                        # The offline/online separation does not hold anymore, so we cannot precompute 
                        # reduced operators.
                        print("... skipped due to inefficient evaluation")
                    
                #  @}
                ########################### end - OFFLINE STAGE - end ########################### 
                
                ###########################     PROBLEM SPECIFIC     ########################### 
                ## @defgroup ProblemSpecific Problem specific methods
                #  @{
                    
                ## Assemble operators for error estimation
                def assemble_error_estimation_operators(self, term):
                    if self.current_stage == "online": # *cannot* load from file
                        # The offline/online separation does not hold anymore, so we need to re-assemble operators,
                        # because the assemble_error_estimation_operators() of the truth problem *may* 
                        # return parameter dependent operators.
                        # Thus, call the parent method enforcing current_stage = "offline"
                        self.current_stage = "offline" # temporary change
                        output = ReducedParametrizedProblem_DerivedClass.assemble_error_estimation_operators(self, term)
                        self.current_stage = "online" # revert temporary change
                        # Return
                        return output
                    else:
                        # Call parent method
                        return ReducedParametrizedProblem_DerivedClass.assemble_error_estimation_operators(self, term)
                                
                #  @}
                ########################### end - PROBLEM SPECIFIC - end ########################### 
                    
            return _AlsoDecorateErrorEstimationOperators_Class
        else:
            return ReducedParametrizedProblem_DecoratedClass
           
    @_AlsoDecorateErrorEstimationOperators
    class ExactParametrizedFunctionEvaluationDecoratedReducedProblem_Class(ReducedParametrizedProblem_DerivedClass):
        ## Default initialization of members
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReducedParametrizedProblem_DerivedClass.__init__(self, truth_problem)
            # Avoid useless assemblies
            self._solve.__func__.previous_mu = None
            self._solve.__func__.previous_self_N = None
        
        ###########################     ONLINE STAGE     ########################### 
        ## @defgroup OnlineStage Methods related to the online stage
        #  @{
    
        ## Initialize data structures required for the online phase
        def init(self, current_stage="online"):
            if current_stage == "online":
                # The offline/online separation does not hold anymore, so in assemble_operator()
                # we need to re-assemble operators. Their data structure is initialized (with
                # suitable size) in this initialization method.
                for term in self.truth_problem.operator:
                    self.operator[term] = AffineExpansionOnlineStorage(self.truth_problem.Q[term])
                    # Make sure to disable the save() method of the operator, which is 
                    # called internally by assemble_operator() since it is not possible
                    # to precompute operators, and thus they should not be saved
                    def disabled_save(self, folder, filename):
                        pass
                    self.operator[term].save = types.MethodType(disabled_save, self.operator[term])
                    # Make sure to raise an error if the load() method of the operator,
                    # since we have not saved anything and it should never be called
                    def error_load(self, folder, filename):
                        raise AttributeError
                    self.operator[term].load = types.MethodType(error_load, self.operator[term])
                # Then, call parent method, which in turn will call assemble_operator()
                # Note that some of the assembled operators will be overwritten at
                # each solve() call
                ReducedParametrizedProblem_DerivedClass.init(self, current_stage)
            else:
                # Call parent method
                ReducedParametrizedProblem_DerivedClass.init(self, current_stage)
                
        # Perform an online solve (internal)
        def _solve(self, N):
            # The offline/online separation does not hold anymore, so, similarly to what we did in
            # the truth problem, also at the reduced-order level we need to re-assemble operators,
            # because the assemble_operator() *may* return parameter dependent operators.
            if self._solve.__func__.previous_mu != self.mu or self._solve.__func__.previous_self_N != self.N:
                if self._solve.__func__.previous_mu != self.mu: # re-assemble truth operators
                    assert self.truth_problem.mu == self.mu
                    self.truth_problem.init()
                current_stage_is_offline = (self.current_stage == "offline")
                if current_stage_is_offline:
                    self.current_stage = "online" # temporary change
                self.build_reduced_operators()
                if current_stage_is_offline:
                    self.current_stage = "offline" # revert temporary change
                # Avoid useless assemblies
                self._solve.__func__.previous_mu = self.mu
                self._solve.__func__.previous_self_N = self.N
            return ReducedParametrizedProblem_DerivedClass._solve(self, N)
    
        #  @}
        ########################### end - ONLINE STAGE - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
            
        ## Assemble the reduced order affine expansion.
        def build_reduced_operators(self):
            if self.current_stage == "online":
                print("build reduced operators (due to inefficient evaluation)")
                self.current_stage = "offline" # temporary change
                output = ReducedParametrizedProblem_DerivedClass.build_reduced_operators(self)
                self.current_stage = "online" # revert temporary change
            else:
                # The offline/online separation does not hold anymore, so we cannot precompute 
                # reduced operators.
                print("... skipped due to inefficient evaluation")
            
        #  @}
        ########################### end - OFFLINE STAGE - end ########################### 
        
        ###########################     PROBLEM SPECIFIC     ########################### 
        ## @defgroup ProblemSpecific Problem specific methods
        #  @{
            
        ## Assemble the reduced order affine expansion
        def assemble_operator(self, term):
            if self.current_stage == "online": # *cannot* load from file
                # The offline/online separation does not hold anymore, so we need to re-assemble operators,
                # because the assemble_operator() of the truth problem *may* return parameter dependent operators.
                # Thus, call the parent method enforcing current_stage = "offline"
                self.current_stage = "offline" # temporary change
                output = ReducedParametrizedProblem_DerivedClass.assemble_operator(self, term)
                self.current_stage = "online" # revert temporary change
                # Return
                return output
            else:
                # Call parent method
                return ReducedParametrizedProblem_DerivedClass.assemble_operator(self, term)
                
        #  @}
        ########################### end - PROBLEM SPECIFIC - end ########################### 
        
    # return value (a class) for the decorator
    return ExactParametrizedFunctionEvaluationDecoratedReducedProblem_Class
