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

import os
from rbnics.utils.decorators import PreserveClassName, ProblemDecoratorFor
from rbnics.scm.problems.scm_approximation import SCMApproximation
from rbnics.scm.problems.decorated_problem_with_stability_factor_evaluation import DecoratedProblemWithStabilityFactorEvaluation

def SCMDecoratedProblem(
    **decorator_kwargs
):
    from rbnics.scm.problems.exact_stability_factor import ExactStabilityFactor
    from rbnics.scm.problems.scm import SCM
    
    @ProblemDecoratorFor(
        SCM,
        ExactAlgorithm=ExactStabilityFactor
    )
    def SCMDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        
        @DecoratedProblemWithStabilityFactorEvaluation
        @PreserveClassName
        class SCMDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
            # Default initialization of members
            def __init__(self, V, **kwargs):
                # Eigen solver parameters (to be initialized before calling parent initialization as it may
                # update them).
                self._eigen_solver_parameters = {
                    "bounding_box_minimum": dict(),
                    "bounding_box_maximum": dict(),
                    "stability_factor": dict()
                }
                # Additional function space required by stability factor computations (to be initialized before
                # calling parent initialization as it will update them)
                self.stability_factor_V = None
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Additional terms required by stability factor computations
                self.terms.extend(["stability_factor_left_hand_matrix", "stability_factor_right_hand_matrix"])
                self.terms_order.update({"stability_factor_left_hand_matrix": 2, "stability_factor_right_hand_matrix": 2})
                # Storage for SCM reduced problems
                self.SCM_approximation = SCMApproximation(self, os.path.join(self.name(), "scm"))
                # Stability factor eigen problem
                self.stability_factor_calculator = self.SCM_approximation.stability_factor_calculator
                self.stability_factor_lower_bound_calculator = self.SCM_approximation
                
        # return value (a class) for the decorator
        return SCMDecoratedProblem_Class
    
    # return the decorator itself
    return SCMDecoratedProblem_Decorator
