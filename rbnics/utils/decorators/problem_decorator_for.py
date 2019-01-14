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

from rbnics.utils.decorators.preserve_class_name import PreserveClassName
from rbnics.utils.decorators.store_problem_decorators_for_factories import StoreProblemDecoratorsForFactories

def ProblemDecoratorFor(Algorithm, ExactAlgorithm=None, replaces=None, replaces_if=None, **kwargs):
    def ProblemDecoratorFor_Decorator(ProblemDecorator):
        def ProblemDecorator_WithStorage(Problem):
            @StoreProblemDecoratorsForFactories(Problem, Algorithm, ExactAlgorithm, **kwargs)
            @PreserveClassName
            class DecoratedProblem(ProblemDecorator(Problem)):
                pass
            
            # Return
            return DecoratedProblem
        # Done with the storage, return the new problem decorator
        return ProblemDecorator_WithStorage
    return ProblemDecoratorFor_Decorator
