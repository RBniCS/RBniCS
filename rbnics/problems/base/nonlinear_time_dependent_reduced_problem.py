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

from rbnics.backends import assign
from rbnics.problems.base.time_dependent_reduced_problem import TimeDependentReducedProblem
from rbnics.utils.decorators import Extends, override

def NonlinearTimeDependentReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True)
    #@NonlinearReducedProblem # this is usually already applied to parent, since we first create a problem class for the steady case
    @TimeDependentReducedProblem
    class NonlinearTimeDependentReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        
        # Store solution dot while solving the nonlinear problem
        def _store_solution_dot(self, solution_dot):
            assign(self._solution_dot, solution_dot)
        
    # return value (a class) for the decorator
    return NonlinearTimeDependentReducedProblem_Class
    
