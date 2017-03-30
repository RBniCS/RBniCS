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
## @file elliptic_coercive_reduced_problem.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.problems.base.time_dependent_rb_reduced_problem import TimeDependentRBReducedProblem
from rbnics.utils.decorators import Extends

def NonlinearTimeDependentRBReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True)
    #@NonlinearRBReducedProblem # this is usually already applied to parent, since we first create a problem class for the steady case
    @TimeDependentRBReducedProblem
    class NonlinearTimeDependentRBReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        pass
                
    # return value (a class) for the decorator
    return NonlinearTimeDependentRBReducedProblem_Class
    
