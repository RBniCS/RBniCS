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
## @file scm.py
#  @brief Implementation of the successive constraints method for the approximation of the coercivity constant
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.problems.base import ParametrizedProblem
from RBniCS.utils.decorators import Extends, override, ProblemDecoratorFor
from RBniCS.scm.problems.parametrized_hermitian_eigenproblem import ParametrizedHermitianEigenProblem
from RBniCS.scm.problems.scm_decorated_problem import SCM

def ExactCoercivityConstantDecoratedProblem(
    constrain_minimum_eigenvalue = 1.e5,
    eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.e-5),
    **decorator_kwargs
):
    @ProblemDecoratorFor(ExactCoercivityConstant, replaces=SCM,
        constrain_minimum_eigenvalue = constrain_minimum_eigenvalue,
        eigensolver_parameters = eigensolver_parameters
    )
    def ExactCoercivityConstantDecoratedProblem_Decorator(ParametrizedProblem_DerivedClass):
        
        @Extends(ParametrizedProblem_DerivedClass, preserve_class_name=True)
        class ExactCoercivityConstantDecoratedProblem_Class(ParametrizedProblem_DerivedClass):
            ## Default initialization of members
            @override
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedProblem_DerivedClass.__init__(self, V, **kwargs)
                
                self.exact_coercivity_constant_calculator = ParametrizedHermitianEigenProblem(self, "a", True, constrain_minimum_eigenvalue, "smallest", eigensolver_parameters)
                
            ## Initialize data structures required for the online phase
            @override
            def init(self):
                # Call to Parent
                ParametrizedProblem_DerivedClass.init(self)
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
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
ExactCoercivityConstant = ExactCoercivityConstantDecoratedProblem
    
