# Copyright (C) 2015-2019 by the RBniCS authors
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

from math import sqrt
from numpy import isclose
from rbnics.utils.decorators import PreserveClassName

def DecoratedProblemWithStabilityFactorEvaluation(ParametrizedDifferentialProblem_DerivedClass):
    from rbnics.problems.elliptic import EllipticCoerciveProblem, EllipticProblem
    from rbnics.problems.stokes import StokesProblem
    
    @PreserveClassName
    class DecoratedProblemWithStabilityFactorEvaluation_Class_Base(ParametrizedDifferentialProblem_DerivedClass):
        def evaluate_stability_factor(self):
            (minimum_eigenvalue, _) = self.stability_factor_calculator.solve()
            return minimum_eigenvalue
            
        def get_stability_factor_lower_bound(self):
            if self.stability_factor_lower_bound_calculator is not None: # SCM case
                return self.stability_factor_lower_bound_calculator.get_stability_factor_lower_bound()
            else: # ExactStabilityFactor case: note that we cannot simply call evaluate_stability_factor otherwise the sqrt in the non-coercive cases would get applied twice
                (minimum_eigenvalue, _) = self.stability_factor_calculator.solve()
                return minimum_eigenvalue
            
    # Elliptic coercive problem specialization
    if issubclass(ParametrizedDifferentialProblem_DerivedClass, EllipticCoerciveProblem):
        DecoratedProblemWithStabilityFactorEvaluation_Class = DecoratedProblemWithStabilityFactorEvaluation_Class_Base
        
    # Elliptic non-coercive (needs to be after the coercive case) or Stokes problem specialization
    elif issubclass(ParametrizedDifferentialProblem_DerivedClass, (EllipticProblem, StokesProblem)):
        @PreserveClassName
        class DecoratedProblemWithStabilityFactorEvaluation_Class(DecoratedProblemWithStabilityFactorEvaluation_Class_Base):
            def evaluate_stability_factor(self):
                beta_squared = DecoratedProblemWithStabilityFactorEvaluation_Class_Base.evaluate_stability_factor(self)
                assert beta_squared > 0. or isclose(beta_squared, 0.)
                return sqrt(abs(beta_squared))
                
            def get_stability_factor_lower_bound(self):
                beta_lower_bound_squared = DecoratedProblemWithStabilityFactorEvaluation_Class_Base.get_stability_factor_lower_bound(self)
                assert beta_lower_bound_squared > 0. or isclose(beta_lower_bound_squared, 0.)
                return sqrt(abs(beta_lower_bound_squared))
                
    # Unhandled case
    else:
        DecoratedProblemWithStabilityFactorEvaluation_Class = DecoratedProblemWithStabilityFactorEvaluation_Class_Base
        
    # return value (a class) for the decorator
    return DecoratedProblemWithStabilityFactorEvaluation_Class
