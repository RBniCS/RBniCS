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

def StoreProblemDecoratorsForFactories(Problem, Algorithm, ExactAlgorithm=None, **kwargs):
    def StoreProblemDecoratorsForFactories_Decorator(DecoratedProblem_Base):
        assert issubclass(DecoratedProblem_Base, Problem)
        
        if hasattr(Problem, "UndecoratedProblemClass"):
            UndecoratedProblemClass = Problem.UndecoratedProblemClass
        else:
            UndecoratedProblemClass = Problem
        if hasattr(Problem, "ProblemDecorators"):
            ProblemDecorators = Problem.ProblemDecorators
        else:
            ProblemDecorators = list()
        if hasattr(Problem, "ProblemDecoratorsKwargs"):
            ProblemDecoratorsKwargs = Problem.ProblemDecoratorsKwargs
        else:
            ProblemDecoratorsKwargs = list()
        if hasattr(Problem, "ProblemExactDecorators"):
            ProblemExactDecorators = Problem.ProblemExactDecorators
        else:
            ProblemExactDecorators = list()
        
        # Also store **kwargs as passed to init
        @PreserveClassName
        class DecoratedProblem(DecoratedProblem_Base):
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                DecoratedProblem_Base.__init__(self, V, **kwargs)
                # Store **kwargs
                self.problem_kwargs = kwargs
                
        # Move attributes from the base class to the decorated class
        DecoratedProblem.UndecoratedProblemClass = UndecoratedProblemClass
        # if hasattr(Problem, "UndecoratedProblemClass"):
        #     delattr(Problem, "UndecoratedProblemClass")
        DecoratedProblem.ProblemDecorators = ProblemDecorators
        # if hasattr(Problem, "ProblemDecorators"):
        #     delattr(Problem, "ProblemDecorators")
        DecoratedProblem.ProblemDecoratorsKwargs = ProblemDecoratorsKwargs
        # if hasattr(Problem, "ProblemDecoratorsKwargs"):
        #     delattr(Problem, "ProblemDecoratorsKwargs")
        DecoratedProblem.ProblemExactDecorators = ProblemExactDecorators
        # if hasattr(Problem, "ProblemExactDecorators"):
        #     delattr(Problem, "ProblemExactDecorators")
        # ... and append the new problem decorator
        if Algorithm in DecoratedProblem.ProblemDecorators:
            assert kwargs in DecoratedProblem.ProblemDecoratorsKwargs, "You have decorated twice the problem with same decorator but different kwargs"
            assert ExactAlgorithm in DecoratedProblem.ProblemExactDecorators, "You have decorated twice the problem with same decorator but different exact decorator"
        else:
            DecoratedProblem.ProblemDecorators.append(Algorithm)
            DecoratedProblem.ProblemDecoratorsKwargs.append(kwargs)
            DecoratedProblem.ProblemExactDecorators.append(ExactAlgorithm)
        
        # Return
        return DecoratedProblem
    # Return
    return StoreProblemDecoratorsForFactories_Decorator
