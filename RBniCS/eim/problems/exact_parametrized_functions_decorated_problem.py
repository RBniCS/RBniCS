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
## @file eim.py
#  @brief Implementation of the empirical interpolation method for the interpolation of parametrized functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators import Extends, override, ProblemDecoratorFor

def ExactParametrizedFunctionsDecoratedProblem(**decorator_kwargs):

    from RBniCS.eim.problems.deim import DEIM
    from RBniCS.eim.problems.eim import EIM
    from RBniCS.eim.problems.exact_parametrized_functions import ExactParametrizedFunctions
    
    @ProblemDecoratorFor(ExactParametrizedFunctions, replaces=(DEIM, EIM))
    def ExactParametrizedFunctionsDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        
        @Extends(ParametrizedDifferentialProblem_DerivedClass, preserve_class_name=True)
        class ExactParametrizedFunctionsDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
            
            ## Default initialization of members
            @override
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Avoid useless assemblies
                self._solve__previous_mu = None
            
            ###########################     OFFLINE STAGE     ########################### 
            ## @defgroup OfflineStage Methods related to the offline stage
            #  @{
            
            ## Perform a truth solve
            @override
            def solve(self, **kwargs):
                # The offline/online separation does not hold anymore, so we need to re-assemble operators,
                # because the assemble_operator() *may* return parameter dependent operators.
                if self._solve__previous_mu != self.mu:
                    self.init()
                    # Avoid useless assemblies
                    self._solve__previous_mu = self.mu
                return ParametrizedDifferentialProblem_DerivedClass.solve(self)
            
            #  @}
            ########################### end - OFFLINE STAGE - end ########################### 
            
        # return value (a class) for the decorator
        return ExactParametrizedFunctionsDecoratedProblem_Class
        
    # return the decorator itself
    return ExactParametrizedFunctionsDecoratedProblem_Decorator
