# Copyright (C) 2015-2018 by the RBniCS authors
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

import os
from rbnics.utils.decorators import PreserveClassName, ProblemDecoratorFor
from rbnics.scm.problems.parametrized_stability_factor_eigenproblem import ParametrizedStabilityFactorEigenProblem

def ExactStabilityFactorDecoratedProblem(
    eigensolver_parameters=None,
    **decorator_kwargs
):
    if eigensolver_parameters is None:
        eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.e-5)
        
    from rbnics.scm.problems.exact_stability_factor import ExactStabilityFactor
    
    @ProblemDecoratorFor(ExactStabilityFactor, eigensolver_parameters=eigensolver_parameters)
    def ExactStabilityFactorDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        
        @PreserveClassName
        class ExactStabilityFactorDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
            # Default initialization of members
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                
                self.exact_stability_factor_calculator = ParametrizedStabilityFactorEigenProblem(self, "a", True, "smallest", eigensolver_parameters, os.path.join(self.name(), "exact_stability_factor"))
                
            # Initialize data structures required for the online phase
            def init(self):
                # Call to Parent
                ParametrizedDifferentialProblem_DerivedClass.init(self)
                # Init exact stability factor computations
                self.exact_stability_factor_calculator.init()
            
            # Return the lower bound for the stability factor.
            def get_stability_factor(self):
                (minimum_eigenvalue, _) = self.exact_stability_factor_calculator.solve()
                return minimum_eigenvalue
                
        # return value (a class) for the decorator
        return ExactStabilityFactorDecoratedProblem_Class
        
    # return the decorator itself
    return ExactStabilityFactorDecoratedProblem_Decorator
