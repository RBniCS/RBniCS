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

def ExactCoercivityConstantDecoratedReducedProblem(ReducedParametrizedProblem_DerivedClass):
    
    class ExactCoercivityConstantDecoratedReducedProblem_Class(ReducedParametrizedProblem_DerivedClass):
        ## Default initialization of members
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReducedParametrizedProblem_DerivedClass.__init__(self, truth_problem)
        
    # return value (a class) for the decorator
    return ExactCoercivityConstantDecoratedReducedProblem_Class
