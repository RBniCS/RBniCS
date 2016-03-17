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
from numpy import log, exp, mean, sqrt # for error analysis
import os # for path and makedir
import shutil # for rm
import random # to randomize selection in case of equal error bound
from RBniCS.parametrized_problem import ParametrizedProblem

def ExactParametrizedFunctionEvaluationDecoratedProblem(*parametrized_expressions):
    def ExactParametrizedFunctionEvaluationDecoratedProblem_Decorator(ParametrizedProblem_DerivedClass):
    
        class ExactParametrizedFunctionEvaluationDecoratedProblem_Class(ParametrizedProblem_DerivedClass):
            ## Default initialization of members
            def __init__(self, V, *args):
                # Call the parent initialization
                ParametrizedProblem_DerivedClass.__init__(self, V, *args)
                # Attach parametrized function objects
                self.parametrized_expressions__as_strings = []
                self.parametrized_expressions = []
                for parametrized_expression__as_string in parametrized_expressions:
                    self.parametrized_expressions__as_strings.append(parametrized_expression__as_string)
                    self.parametrized_expressions.append(ParametrizedExpression())
            
            ###########################     SETTERS     ########################### 
            ## @defgroup Setters Set properties of the reduced order approximation
            #  @{
        
            # Propagate the values of all setters also to the parametrized expressions objects
                                
            ## OFFLINE/ONLINE: set the current value of the parameter
            def setmu(self, mu):
                ParametrizedProblem.setmu(self, mu)
                for i in len(self.parametrized_expressions):
                    self.parametrized_expressions[i].setmu(mu)
                
            #  @}
            ########################### end - SETTERS - end ########################### 
            
            ###########################     OFFLINE STAGE     ########################### 
            ## @defgroup OfflineStage Methods related to the offline stage
            #  @{
            
            ## Initialize data structures required for the offline phase
            def init(self):
                ParametrizedProblem_DerivedClass.init(self)
                # Initialize the parametrized expressions
                for i in len(self.parametrized_expressions):
                    self.parametrized_expressions[i] = ParametrizedExpression(self.parametrized_expressions__as_strings[i], mu=self.mu, element=self.V.ufl_element())
                    
            ## Perform a truth solve
            def solve(self):
                # The offline/online separation does not hold anymore, so we need to re-assemble operators,
                # because the assemble_operator() *may* return parameter dependent operators.
                for term in self.operator:
                    self.operator[term] = AffineExpansionOfflineStorage(self.assemble_operator(term))
                return ParametrizedProblem_DerivedClass.solve(self)
            
            #  @}
            ########################### end - OFFLINE STAGE - end ########################### 
            
        # return value (a class) for the decorator
        return ExactParametrizedFunctionEvaluationDecoratedProblem_Class
    
    # return the decorator itself
    return ExactParametrizedFunctionEvaluationDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
ExactParametrizedFunctionEvaluation = ExactParametrizedFunctionEvaluationDecoratedProblem
