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

from rbnics.problems.base import ParametrizedProblem
from rbnics.utils.decorators import Extends, override, ProblemDecoratorFor
from rbnics.scm.problems.parametrized_coercivity_constant_eigenproblem import ParametrizedCoercivityConstantEigenProblem

def ExactCoercivityConstantDecoratedProblem(
    eigensolver_parameters = None,
    **decorator_kwargs
):
    if eigensolver_parameters is None:
        eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.e-5)
        
    from rbnics.scm.problems.exact_coercivity_constant import ExactCoercivityConstant
    from rbnics.scm.problems.scm import SCM
    
    @ProblemDecoratorFor(ExactCoercivityConstant, replaces=SCM,
        eigensolver_parameters = eigensolver_parameters
    )
    def ExactCoercivityConstantDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        
        @Extends(ParametrizedDifferentialProblem_DerivedClass, preserve_class_name=True)
        class ExactCoercivityConstantDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
            ## Default initialization of members
            @override
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                
                self.exact_coercivity_constant_calculator = ParametrizedCoercivityConstantEigenProblem(self, "a", True, "smallest", eigensolver_parameters)
                
            ## Initialize data structures required for the online phase
            @override
            def init(self):
                # Call to Parent
                ParametrizedDifferentialProblem_DerivedClass.init(self)
                # Init exact coercivity constant computations
                self.exact_coercivity_constant_calculator.init()
            
            ## Return the alpha_lower bound.
            @override
            def get_stability_factor(self):
                (minimum_eigenvalue, _) = self.exact_coercivity_constant_calculator.solve()
                return minimum_eigenvalue
                
                        
        # return value (a class) for the decorator
        return ExactCoercivityConstantDecoratedProblem_Class
        
    # return the decorator itself
    return ExactCoercivityConstantDecoratedProblem_Decorator    
