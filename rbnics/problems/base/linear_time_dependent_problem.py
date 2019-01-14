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

from rbnics.problems.base.linear_problem import LinearProblem
from rbnics.problems.base.time_dependent_problem import TimeDependentProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators

@RequiredBaseDecorators(LinearProblem, TimeDependentProblem)
def LinearTimeDependentProblem(ParametrizedDifferentialProblem_DerivedClass):
    
    @PreserveClassName
    class LinearTimeDependentProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
        def __init__(self, V, **kwargs):
            # Call the parent initialization
            ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
            # Set the problem type in time stepping parameters
            self._time_stepping_parameters["problem_type"] = "linear"
            
    # return value (a class) for the decorator
    return LinearTimeDependentProblem_Class
