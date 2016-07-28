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
from RBniCS.io_utils import print

def ExactParametrizedFunctionEvaluationDecoratedReducedProblem(ReducedParametrizedProblem_DerivedClass):
    
    def _AlsoDecorateErrorEstimationOperators(ReducedParametrizedProblem_DecoratedClass):
        if hasattr(ReducedParametrizedProblem_DecoratedClass, "assemble_error_estimation_operators"):
            class _AlsoDecorateErrorEstimationOperators_Class(ReducedParametrizedProblem_DecoratedClass):
                def __init__(self, truth_problem):
                    # Call the parent initialization
                    ReducedParametrizedProblem_DecoratedClass.__init__(self, truth_problem)
                    # Avoid useless assemblies
                    self.estimate_error.__func__.previous_mu = None
                    self.estimate_error.__func__.previous_self_N = None
                    
                ###########################     ONLINE STAGE     ########################### 
                ## @defgroup OnlineStage Methods related to the online stage
                #  @{
            
                ## Initialize data structures required for the online phase
                def init(self, current_stage="online"):
                    ReducedParametrizedProblem_DecoratedClass.init(self, current_stage)
                    # The offline/online separation does not hold anymore, so in 
                    # assemble_error_estimation_operators() we need to re-assemble operators. 
                    # Thus, for any value of current_stage, we initialize error estimation
                    # operators of the reduced problem as if we were offline
                    ReducedParametrizedProblem_DecoratedClass._init_error_estimation_operators(self, "offline")
                    for term1 in self.truth_problem.Q:
                        for term2 in self.truth_problem.Q:
                            if term1 > term2: # alphabetical order
                                continue
                            
                            self._disable_load_and_save_for_online_storage(self.riesz_product[term1 + term2], self.folder["error_estimation"])
                                                    
                ## Return the numerator of the error bound for the current solution
                def estimate_error(self):
                    # The offline/online separation does not hold anymore, so, similarly to what we did in
                    # the truth problem, also at the reduced-order level we need to re-assemble operators,
                    # because the assemble_operator() *may* return parameter dependent operators.
                    assert(self._solve.__func__.previous_mu == self.mu) # estimate_error is always called after _solve
                    if self.estimate_error.__func__.previous_mu != self.mu or self.estimate_error.__func__.previous_self_N != self.N:
                        self.build_error_estimation_operators("online")
                        # Avoid useless assemblies
                        self.estimate_error.__func__.previous_mu = self.mu
                        self.estimate_error.__func__.previous_self_N = self.N
                    return ReducedParametrizedProblem_DecoratedClass.estimate_error(self)
                    
                ## Return the numerator of the error bound for the current output
                def estimate_error_output(self):
                    # The offline/online separation does not hold anymore, so, similarly to what we did in
                    # the truth problem, also at the reduced-order level we need to re-assemble operators,
                    # because the assemble_operator() *may* return parameter dependent operators.
                    assert(self._solve.__func__.previous_mu == self.mu) # estimate_error is always called after _solve
                    if self.estimate_error.__func__.previous_mu != self.mu or self.estimate_error.__func__.previous_self_N != self.N:
                        self.build_error_estimation_operators("online")
                        # Avoid useless assemblies
                        self.estimate_error.__func__.previous_mu = self.mu
                        self.estimate_error.__func__.previous_self_N = self.N
                        # Note that we use the the same cache as estimate_error, since (at least part of)
                        # error estimation operators is used by both methods
                    return ReducedParametrizedProblem_DecoratedClass.estimate_error_output(self)
                        
                #  @}
                ########################### end - ONLINE STAGE - end ########################### 
                
                ###########################     OFFLINE STAGE     ########################### 
                ## @defgroup OfflineStage Methods related to the offline stage
                #  @{
                    
                ## Build operators for error estimation
                def build_error_estimation_operators(self, current_stage="offline"):
                    def log(string):
                        from dolfin import log, PROGRESS
                        log(PROGRESS, string)
                    if current_stage == "online":
                        log("build operators for error estimation (due to inefficient evaluation)")
                        for term in self.truth_problem.Q:
                            for q in range(self.truth_problem.Q[term]):
                                self.riesz[term][q].clear()
                        self.build_error_estimation_operators.__func__.initialized = False
                        ReducedParametrizedProblem_DecoratedClass.build_error_estimation_operators(self)
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
                def assemble_error_estimation_operators(self, term, current_stage="online"):
                    if current_stage == "online": # *cannot* load from file
                        # The offline/online separation does not hold anymore, so we need to re-assemble operators,
                        # because the assemble_error_estimation_operators() of the truth problem *may* 
                        # return parameter dependent operators.
                        # Thus, call the parent method enforcing current_stage = "offline"
                        output = ReducedParametrizedProblem_DecoratedClass.assemble_error_estimation_operators(self, term, "offline")
                        # Return
                        return output
                    else:
                        # Call parent method
                        return ReducedParametrizedProblem_DecoratedClass.assemble_error_estimation_operators(self, term, current_stage)
                                
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
            self._init_basis_functions(current_stage)
            # The offline/online separation does not hold anymore, so in assemble_operator()
            # we need to re-assemble operators. Thus, for any value of current_stage,
            # we initialize the operators of the reduced problem as if we were offline
            self._init_operators("offline")
            for term in self.terms:
                self._disable_load_and_save_for_online_storage(self.operator[term], self.folder["reduced_operators"])

        
        def _disable_load_and_save_for_online_storage(self, online_storage, folder):
            # Make sure to disable the save() method of the operator, which is 
            # called internally by assemble_operator() since it is not possible
            # to precompute operators, and thus they should not be saved
            def disabled_save(self, folder, filename):
                pass
            online_storage.save = types.MethodType(disabled_save, online_storage)
            # Make sure to raise an error if the load() method of the operator,
            # since we have not saved anything and it should never be called
            def error_load(self, folder, filename):
                raise AttributeError("Cannot load from file due to inefficient evaluation")
            online_storage.load = types.MethodType(error_load, online_storage)
            # However, write a dummy file to make sure that restart is enabled
            folder.touch_file("disabled_due_to_inefficient_evaluation")
            
        
        # Perform an online solve (internal)
        def _solve(self, N):
            # The offline/online separation does not hold anymore, so, similarly to what we did in
            # the truth problem, also at the reduced-order level we need to re-assemble operators,
            # because the assemble_operator() *may* return parameter dependent operators.
            if self._solve.__func__.previous_mu != self.mu or self._solve.__func__.previous_self_N != self.N:
                if self._solve.__func__.previous_mu != self.mu: # re-assemble truth operators
                    assert self.truth_problem.mu == self.mu
                    self.truth_problem.init()
                self.build_reduced_operators("online")
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
        def build_reduced_operators(self, current_stage="offline"):
            def log(string):
                from dolfin import log, PROGRESS
                log(PROGRESS, string)
            if current_stage == "online":
                log("build reduced operators (due to inefficient evaluation)")
                output = ReducedParametrizedProblem_DerivedClass.build_reduced_operators(self)
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
        def assemble_operator(self, term, current_stage="online"):
            if current_stage == "online": # *cannot* load from file
                # The offline/online separation does not hold anymore, so we need to re-assemble operators,
                # because the assemble_operator() of the truth problem *may* return parameter dependent operators.
                # Thus, call the parent method enforcing current_stage = "offline"
                output = ReducedParametrizedProblem_DerivedClass.assemble_operator(self, term, "offline")
                # Return
                return output
            else:
                # Call parent method
                return ReducedParametrizedProblem_DerivedClass.assemble_operator(self, term, current_stage)
                
        #  @}
        ########################### end - PROBLEM SPECIFIC - end ########################### 
        
    # return value (a class) for the decorator
    return ExactParametrizedFunctionEvaluationDecoratedReducedProblem_Class
