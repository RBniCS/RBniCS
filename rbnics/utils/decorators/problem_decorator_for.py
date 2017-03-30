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
## @file numpy_io.py
#  @brief I/O helper functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.mpi import log, DEBUG
from RBniCS.utils.decorators import Extends, override

def ProblemDecoratorFor(Algorithm, ExactAlgorithm=None, replaces=None, replaces_if=None, **kwargs):
    def ProblemDecoratorFor_Decorator(ProblemDecorator):
        def ProblemDecorator_WithStorage(Problem):
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
                
            # Apply decorator
            DecoratedProblem_Base = ProblemDecorator(Problem)
            
            # Also store **kwargs as passed to init
            @Extends(DecoratedProblem_Base, preserve_class_name=True)
            class DecoratedProblem(DecoratedProblem_Base):
                @override
                def __init__(self, V, **kwargs):
                    # Call the parent initialization
                    DecoratedProblem_Base.__init__(self, V, **kwargs)
                    # Store **kwargs
                    self.problem_kwargs = kwargs
                    
            # Move attributes from the base class to the decorated class
            DecoratedProblem.UndecoratedProblemClass = UndecoratedProblemClass
            if hasattr(Problem, "UndecoratedProblemClass"):
                delattr(Problem, "UndecoratedProblemClass")
            DecoratedProblem.ProblemDecorators = ProblemDecorators
            if hasattr(Problem, "ProblemDecorators"):
                delattr(Problem, "ProblemDecorators")
            DecoratedProblem.ProblemDecoratorsKwargs = ProblemDecoratorsKwargs
            if hasattr(Problem, "ProblemDecoratorsKwargs"):
                delattr(Problem, "ProblemDecoratorsKwargs")
            DecoratedProblem.ProblemExactDecorators = ProblemExactDecorators
            if hasattr(Problem, "ProblemExactDecorators"):
                delattr(Problem, "ProblemExactDecorators")
            # ... and append the new problem decorator
            DecoratedProblem.ProblemDecorators.append(Algorithm) # replaces and replaces_if are not used, but will be passed also to reduction methods and reduced problem.
            DecoratedProblem.ProblemDecoratorsKwargs.append(kwargs)
            DecoratedProblem.ProblemExactDecorators.append(ExactAlgorithm)
            
            # Return
            return DecoratedProblem
        # Done with the storage, return the new problem decorator
        return ProblemDecorator_WithStorage
    return ProblemDecoratorFor_Decorator
    
