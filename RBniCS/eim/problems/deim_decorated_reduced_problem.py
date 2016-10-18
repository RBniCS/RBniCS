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
## @file eim.py
#  @brief Implementation of the empirical interpolation method for the interpolation of parametrized functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators import Extends, override, ReducedProblemDecoratorFor
from RBniCS.eim.problems.deim_decorated_problem import DEIM

@ReducedProblemDecoratorFor(DEIM)
def DEIMDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class DEIMDecoratedReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        ## Default initialization of members
        @override
        def __init__(self, truth_problem):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem)
            # Store the truth problem which should be updated for DEIM
            self.truth_problem_for_DEIM = self.truth_problem
            # ... this makes sure that, in case self.truth_problem is replaced (because of multilevel reduction)
            #     the correct problem is called for what concerns DEIM computations
            
        @override
        def _solve(self, N, **kwargs):
            self._update_N_DEIM(**kwargs)
            return ParametrizedReducedDifferentialProblem_DerivedClass._solve(self, N, **kwargs)
            
        def _update_N_DEIM(self, **kwargs):
            self.truth_problem_for_DEIM._update_N_DEIM(**kwargs)
            
        @override
        def compute_theta(self, term):
            return self.truth_problem_for_DEIM.compute_theta(term)
        
    # return value (a class) for the decorator
    return DEIMDecoratedReducedProblem_Class
