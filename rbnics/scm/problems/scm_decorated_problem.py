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

from rbnics.utils.decorators import Extends, ProblemDecoratorFor
from rbnics.scm.problems.scm_approximation import SCMApproximation

def SCMDecoratedProblem(
    M_e = None,
    M_p = None,
    bounding_box_minimum_eigensolver_parameters = None,
    bounding_box_maximum_eigensolver_parameters = None,
    coercivity_eigensolver_parameters = None,
    **decorator_kwargs
):
    if bounding_box_minimum_eigensolver_parameters is None:
        bounding_box_minimum_eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.e-5)
    if bounding_box_maximum_eigensolver_parameters is None:
        bounding_box_maximum_eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.e5)
    if coercivity_eigensolver_parameters is None:
        coercivity_eigensolver_parameters = dict(spectral_transform="shift-and-invert", spectral_shift=1.e-5)
    
    from rbnics.scm.problems.exact_coercivity_constant import ExactCoercivityConstant
    from rbnics.scm.problems.scm import SCM
    
    @ProblemDecoratorFor(SCM, ExactAlgorithm=ExactCoercivityConstant,
        M_e = M_e,
        M_p = M_p,
        bounding_box_minimum_eigensolver_parameters = bounding_box_minimum_eigensolver_parameters,
        bounding_box_maximum_eigensolver_parameters = bounding_box_maximum_eigensolver_parameters,
        coercivity_eigensolver_parameters = coercivity_eigensolver_parameters
    )
    def SCMDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
    
        @Extends(ParametrizedDifferentialProblem_DerivedClass, preserve_class_name=True)
        class SCMDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
            ## Default initialization of members
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Store input parameters from the decorator factory in a dict
                decorator_inputs = dict()
                decorator_inputs["M_e"] = M_e
                decorator_inputs["M_p"] = M_p
                decorator_inputs["bounding_box_minimum_eigensolver_parameters"] = bounding_box_minimum_eigensolver_parameters
                decorator_inputs["bounding_box_maximum_eigensolver_parameters"] = bounding_box_maximum_eigensolver_parameters
                decorator_inputs["coercivity_eigensolver_parameters"] = coercivity_eigensolver_parameters
                # Storage for SCM reduced problems
                self.SCM_approximation = SCMApproximation(self, self.name() + "/scm", **decorator_inputs)
                
            ## Return the alpha_lower bound.
            def get_stability_factor(self):
                return self.SCM_approximation.get_stability_factor_lower_bound()

        # return value (a class) for the decorator
        return SCMDecoratedProblem_Class
    
    # return the decorator itself
    return SCMDecoratedProblem_Decorator
