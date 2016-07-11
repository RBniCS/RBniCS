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
from RBniCS.problems import ParametrizedProblem
from RBniCS.linear_algebra import AffineExpansionOfflineStorage

def ExactParametrizedFunctionEvaluationDecoratedProblem(ParametrizedProblem_DerivedClass):

    class ExactParametrizedFunctionEvaluationDecoratedProblem_Class(ParametrizedProblem_DerivedClass):
        
        ## Default initialization of members
        def __init__(self, V, **kwargs):
            # Call the parent initialization
            ParametrizedProblem_DerivedClass.__init__(self, V, **kwargs)
            # Avoid useless assemblies
            self.solve.__func__.previous_mu = None
            
            # Signal to the factory that this problem has been decorated
            if not hasattr(self, "_problem_decorators"):
                self._problem_decorators = dict() # string to bool
            self._problem_decorators["ExactParametrizedFunctionEvaluation"] = True
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
        
        ## Perform a truth solve
        def solve(self):
            # The offline/online separation does not hold anymore, so we need to re-assemble operators,
            # because the assemble_operator() *may* return parameter dependent operators.
            if self.solve.__func__.previous_mu != self.mu:
                self.init()
                # Avoid useless assemblies
                self.solve.__func__.previous_mu = self.mu
            return ParametrizedProblem_DerivedClass.solve(self)
        
        #  @}
        ########################### end - OFFLINE STAGE - end ########################### 
        
        ###########################     I/O     ########################### 
        ## @defgroup IO Input/output methods
        #  @{
                    
        ## Get the name of the problem, to be used as a prefix for output folders.
        # Overridden to use the parent name
        @classmethod
        def name(cls):
            assert len(cls.__bases__) == 1
            return cls.__bases__[0].name()
            
        #  @}
        ########################### end - I/O - end ########################### 
        
    # return value (a class) for the decorator
    return ExactParametrizedFunctionEvaluationDecoratedProblem_Class
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
ExactParametrizedFunctionEvaluation = ExactParametrizedFunctionEvaluationDecoratedProblem
