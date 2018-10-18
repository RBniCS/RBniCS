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

from rbnics.utils.decorators import PreserveClassName

def DecoratedReducedProblemWithStabilityFactorEvaluation(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    @PreserveClassName
    class DecoratedReducedProblemWithStabilityFactorEvaluation_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        # Default initialization of members
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
        # Return the lower bound for the stability factor.
        def get_stability_factor_lower_bound(self, N=None, **kwargs):
            # Call the exact evaluation, since its computational cost is low because we are dealing with
            # left-hand and right-hand side matrices of small dimensions
            return self.evaluate_stability_factor()
                
        def evaluate_stability_factor(self, N=None, **kwargs):
            raise NotImplementedError("Evaluation of reduced stability factor not implemented yet")
        
    # return value (a class) for the decorator
    return DecoratedReducedProblemWithStabilityFactorEvaluation_Class
