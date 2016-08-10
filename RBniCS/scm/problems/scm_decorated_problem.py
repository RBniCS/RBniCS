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

from RBniCS.utils.decorators import Extends, override, ProblemDecoratorFor
from RBniCS.scm.problems.scm_approximation import SCMApproximation

def SCMDecoratedProblem(
    M_e = -1,
    M_p = -1,
    constrain_minimum_eigenvalue = 1.e5,
    constrain_maximum_eigenvalue = 1.e-5,
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
    
    from RBniCS.scm.problems.exact_coercivity_constant_decorated_problem import ExactCoercivityConstant    
    
    @ProblemDecoratorFor(SCM, ExactAlgorithm=ExactCoercivityConstant,
        M_e = M_e,
        M_p = M_p,
        constrain_minimum_eigenvalue = constrain_minimum_eigenvalue,
        constrain_maximum_eigenvalue = constrain_maximum_eigenvalue,
        bounding_box_minimum_eigensolver_parameters = bounding_box_minimum_eigensolver_parameters,
        bounding_box_maximum_eigensolver_parameters = bounding_box_maximum_eigensolver_parameters,
        coercivity_eigensolver_parameters = coercivity_eigensolver_parameters
    )
    def SCMDecoratedProblem_Decorator(ParametrizedProblem_DerivedClass):
    
        @Extends(ParametrizedProblem_DerivedClass, preserve_class_name=True)
        class SCMDecoratedProblem_Class(ParametrizedProblem_DerivedClass):
            ## Default initialization of members
            @override
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedProblem_DerivedClass.__init__(self, V, **kwargs)
                # Store input parameters from the decorator factory in a dict
                decorator_inputs = dict()
                decorator_inputs["M_e"] = M_e
                decorator_inputs["M_p"] = M_p
                decorator_inputs["constrain_minimum_eigenvalue"] = constrain_minimum_eigenvalue
                decorator_inputs["constrain_maximum_eigenvalue"] = constrain_maximum_eigenvalue
                decorator_inputs["bounding_box_minimum_eigensolver_parameters"] = bounding_box_minimum_eigensolver_parameters
                decorator_inputs["bounding_box_maximum_eigensolver_parameters"] = bounding_box_maximum_eigensolver_parameters
                decorator_inputs["coercivity_eigensolver_parameters"] = coercivity_eigensolver_parameters
                # Storage for SCM reduced problems
                self.SCM_approximation = SCMApproximation(self, type(self).__name__ + "/scm", **decorator_inputs)
                
            ## Return the alpha_lower bound.
            @override
            def get_stability_factor(self):
                return self.SCM_approximation.get_stability_factor_lower_bound(self.mu)

        # return value (a class) for the decorator
        return SCMDecoratedProblem_Class
    
    # return the decorator itself
    return SCMDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
SCM = SCMDecoratedProblem
    
