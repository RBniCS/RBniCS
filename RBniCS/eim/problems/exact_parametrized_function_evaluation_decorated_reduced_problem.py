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

import types
from RBniCS.problems import ParametrizedProblem
from RBniCS.linear_algebra import AffineExpansionOnlineStorage

def ExactParametrizedFunctionEvaluationDecoratedReducedProblem(ReducedParametrizedProblem_DerivedClass):

    class ExactParametrizedFunctionEvaluationDecoratedReducedProblem_Class(ReducedParametrizedProblem_DerivedClass):
        ## Default initialization of members
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReducedParametrizedProblem_DerivedClass.__init__(self, truth_problem)
            
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
                    self.Q[term] = self.truth_problem.Q[term]
                    self.operator[term] = AffineExpansionOnlineStorage(self.Q[term])
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
                ReducedParametrizedProblem_DerivedClass.init(self, current_stage)
            else:
                # Call parent method
                ReducedParametrizedProblem_DerivedClass.init(self, current_stage)
                
        # Perform an online solve (internal)
        def _solve(self, N):
            # The offline/online separation does not hold anymore, so, similarly to what we did in
            # the truth problem, also at the reduced-order level we need to re-assemble operators,
            # because the assemble_operator() *may* return parameter dependent operators.
            for term in self.operator:
                self.operator[term] = self.assemble_operator(term)
            return ReducedParametrizedProblem_DerivedClass._solve(self, N)
    
        #  @}
        ########################### end - ONLINE STAGE - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
            
        ## Assemble the reduced order affine expansion.
        def build_reduced_operators(self):
            assert self.current_stage == "offline"
            # The offline/online separation does not hold anymore, so we cannot precompute 
            # reduced operators.
            pass
            
        #  @}
        ########################### end - OFFLINE STAGE - end ########################### 
        
    # return value (a class) for the decorator
    return ExactParametrizedFunctionEvaluationDecoratedReducedProblem_Class
