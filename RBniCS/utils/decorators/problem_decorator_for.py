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
## @file numpy_io.py
#  @brief I/O helper functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.io import log, DEBUG

def ProblemDecoratorFor(Algorithm, replaces=None, replaces_if=None):
    def ProblemDecoratorFor_Decorator(ProblemDecorator):
        def ProblemDecorator_WithStorage(Problem):
            if not hasattr(Problem, "ProblemDecorators"):
                Problem.ProblemDecorators = list()
            Problem.ProblemDecorators.append(Algorithm) # replaces and replaces_if are not used, but will be passed also to reduction methods and reduced problem.
            # Done with the storage, apply the decorator
            return ProblemDecorator(Problem)
        # Done with the storage, return the new problem decorator
        return ProblemDecorator_WithStorage
    return ProblemDecoratorFor_Decorator
    
