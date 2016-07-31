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

from RBniCS.utils.decorators import Extends, override, ProblemDecoratorFor
from RBniCS.eim.problems.eim_decorated_problem import EIM

def ExactParametrizedFunctionsDecoratedProblem():
    @ProblemDecoratorFor(ExactParametrizedFunctions, replaces=(EIM,))
    def ExactParametrizedFunctionsDecoratedProblem_Decorator(ParametrizedProblem_DerivedClass):
        
        @Extends(ParametrizedProblem_DerivedClass, preserve_class_name=True)
        class ExactParametrizedFunctionsDecoratedProblem_Class(ParametrizedProblem_DerivedClass):
            
            ## Default initialization of members
            @override
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedProblem_DerivedClass.__init__(self, V, **kwargs)
                # Avoid useless assemblies
                self.solve.__func__.previous_mu = None
            
            ###########################     OFFLINE STAGE     ########################### 
            ## @defgroup OfflineStage Methods related to the offline stage
            #  @{
            
            ## Perform a truth solve
            @override
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
            
        # return value (a class) for the decorator
        return ExactParametrizedFunctionsDecoratedProblem_Class
        
    # return the decorator itself
    return ExactParametrizedFunctionsDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
ExactParametrizedFunctions = ExactParametrizedFunctionsDecoratedProblem
