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

from rbnics.problems.base.nonlinear_rb_reduced_problem import NonlinearRBReducedProblem
from rbnics.problems.base.time_dependent_rb_reduced_problem import TimeDependentRBReducedProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators

@RequiredBaseDecorators(NonlinearRBReducedProblem, TimeDependentRBReducedProblem)
def NonlinearTimeDependentRBReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    @PreserveClassName
    class NonlinearTimeDependentRBReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        pass
                
    # return value (a class) for the decorator
    return NonlinearTimeDependentRBReducedProblem_Class
