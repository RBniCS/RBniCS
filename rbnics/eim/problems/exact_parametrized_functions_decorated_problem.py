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

from rbnics.backends import SymbolicParameters
from rbnics.utils.decorators import Extends, override, ProblemDecoratorFor

def ExactParametrizedFunctionsDecoratedProblem(**decorator_kwargs):

    from rbnics.eim.problems.deim import DEIM
    from rbnics.eim.problems.eim import EIM
    from rbnics.eim.problems.exact_parametrized_functions import ExactParametrizedFunctions
    
    @ProblemDecoratorFor(ExactParametrizedFunctions, replaces=(DEIM, EIM))
    def ExactParametrizedFunctionsDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        
        @Extends(ParametrizedDifferentialProblem_DerivedClass, preserve_class_name=True)
        class ExactParametrizedFunctionsDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
            
            @override
            def init(self):
                # Temporarily replace float parameters with symbols, so that the forms do not hardcode
                # the current value of the parameter while assemblying.
                mu_float = self.mu
                self.mu = SymbolicParameters(self, self.V, self.mu)
                # Call parent
                output = ParametrizedDifferentialProblem_DerivedClass.init(self)
                # Restore float parameters
                self.mu = mu_float
                # Return
                return output
            
        # return value (a class) for the decorator
        return ExactParametrizedFunctionsDecoratedProblem_Class
        
    # return the decorator itself
    return ExactParametrizedFunctionsDecoratedProblem_Decorator
