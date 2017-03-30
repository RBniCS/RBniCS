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
## @file
#  @brief
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.backends import assign
from rbnics.utils.decorators import Extends, override

def NonlinearReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class NonlinearReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        
        ## Default initialization of members.
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            
            # Nonlinear solver parameters
            self._nonlinear_solver_parameters = dict()
        
        # Store solution while solving the nonlinear problem
        def _store_solution(self, solution):
            assign(self._solution, solution)
        
    # return value (a class) for the decorator
    return NonlinearReducedProblem_Class
    
