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
from RBniCS.utils.mpi import log, print, PROGRESS
from RBniCS.utils.decorators import Extends, override, ReducedProblemDecoratorFor
from RBniCS.eim.problems.eim_decorated_problem import EIM
from RBniCS.eim.problems.exact_parametrized_functions_decorated_problem import ExactParametrizedFunctions

@ReducedProblemDecoratorFor(ExactParametrizedFunctions, replaces=(EIM,))
def ExactParametrizedFunctionsDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    def _AlsoDecorateErrorEstimationOperators(ReducedParametrizedProblem_DecoratedClass):
        if hasattr(ReducedParametrizedProblem_DecoratedClass, "assemble_error_estimation_operators"):
        
            @Extends(ReducedParametrizedProblem_DecoratedClass, preserve_class_name=True)
            class _AlsoDecorateErrorEstimationOperators_Class(ReducedParametrizedProblem_DecoratedClass):
                @override
                def __init__(self, truth_problem, **kwargs):
                    # Call the parent initialization
                    ReducedParametrizedProblem_DecoratedClass.__init__(self, truth_problem, **kwargs)
                    # Avoid useless assemblies
                    self._estimate_error__previous_mu = None
                    self._estimate_error__previous_self_N = None
                    
                ###########################     ONLINE STAGE     ########################### 
                ## @defgroup OnlineStage Methods related to the online stage
                #  @{
            
                ## Initialize data structures required for the online phase
                @override
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
                @override
                def estimate_error(self):
                    # The offline/online separation does not hold anymore, so, similarly to what we did in
                    # the truth problem, also at the reduced-order level we need to re-assemble operators,
                    # because the assemble_operator() *may* return parameter dependent operators.
                    assert(self._solve__previous_mu == self.mu) # estimate_error is always called after _solve
                    if self._estimate_error__previous_mu != self.mu or self._estimate_error__previous_self_N != self.N:
                        self.build_error_estimation_operators("online")
                        # Avoid useless assemblies
                        self._estimate_error__previous_mu = self.mu
                        self._estimate_error__previous_self_N = self.N
                    return ReducedParametrizedProblem_DecoratedClass.estimate_error(self)
                    
                ## Return the numerator of the error bound for the current output
                @override
                def estimate_error_output(self):
                    # The offline/online separation does not hold anymore, so, similarly to what we did in
                    # the truth problem, also at the reduced-order level we need to re-assemble operators,
                    # because the assemble_operator() *may* return parameter dependent operators.
                    assert(self._solve__previous_mu == self.mu) # estimate_error is always called after _solve
                    if self._estimate_error__previous_mu != self.mu or self._estimate_error__previous_self_N != self.N:
                        self.build_error_estimation_operators("online")
                        # Avoid useless assemblies
                        self._estimate_error__previous_mu = self.mu
                        self._estimate_error__previous_self_N = self.N
                        # Note that we use the the same cache as estimate_error, since (at least part of)
                        # error estimation operators is used by both methods
                    return ReducedParametrizedProblem_DecoratedClass.estimate_error_output(self)
                        
                #  @}
                ########################### end - ONLINE STAGE - end ########################### 
                
                ###########################     OFFLINE STAGE     ########################### 
                ## @defgroup OfflineStage Methods related to the offline stage
                #  @{
                    
                ## Build operators for error estimation
                @override
                def build_error_estimation_operators(self, current_stage="offline"):
                    if current_stage == "online":
                        log(PROGRESS, "build operators for error estimation (due to inefficient evaluation)")
                        for term in self.truth_problem.Q:
                            for q in range(self.truth_problem.Q[term]):
                                self.riesz[term][q].clear()
                        self.build_error_estimation_operators__initialized = False
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
                @override
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
           
    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True) # needs to be first in order to override for last the methods
    @_AlsoDecorateErrorEstimationOperators
    class ExactParametrizedFunctionsDecoratedReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        ## Default initialization of members
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            # Avoid useless assemblies
            self._solve__previous_mu = None
            self._solve__previous_self_N = None
        
        ###########################     ONLINE STAGE     ########################### 
        ## @defgroup OnlineStage Methods related to the online stage
        #  @{
    
        ## Initialize data structures required for the online phase
        @override
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
        @override
        def _solve(self, N, **kwargs):
            # The offline/online separation does not hold anymore, so, similarly to what we did in
            # the truth problem, also at the reduced-order level we need to re-assemble operators,
            # because the assemble_operator() *may* return parameter dependent operators.
            if self._solve__previous_mu != self.mu or self._solve__previous_self_N != self.N:
                if self._solve__previous_mu != self.mu: # re-assemble truth operators
                    assert self.truth_problem.mu == self.mu
                    self.truth_problem.init()
                self.build_reduced_operators("online")
                # Avoid useless assemblies
                self._solve__previous_mu = self.mu
                self._solve__previous_self_N = self.N
            return ParametrizedReducedDifferentialProblem_DerivedClass._solve(self, N, **kwargs)
    
        #  @}
        ########################### end - ONLINE STAGE - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
            
        ## Assemble the reduced order affine expansion.
        @override
        def build_reduced_operators(self, current_stage="offline"):
            if current_stage == "online":
                log(PROGRESS, "build reduced operators (due to inefficient evaluation)")
                output = ParametrizedReducedDifferentialProblem_DerivedClass.build_reduced_operators(self)
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
        @override
        def assemble_operator(self, term, current_stage="online"):
            if current_stage == "online": # *cannot* load from file
                # The offline/online separation does not hold anymore, so we need to re-assemble operators,
                # because the assemble_operator() of the truth problem *may* return parameter dependent operators.
                # Thus, call the parent method enforcing current_stage = "offline"
                output = ParametrizedReducedDifferentialProblem_DerivedClass.assemble_operator(self, term, "offline")
                # Return
                return output
            else:
                # Call parent method
                return ParametrizedReducedDifferentialProblem_DerivedClass.assemble_operator(self, term, current_stage)
                
        #  @}
        ########################### end - PROBLEM SPECIFIC - end ########################### 
        
    # return value (a class) for the decorator
    return ExactParametrizedFunctionsDecoratedReducedProblem_Class
