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

from RBniCS.utils.decorators.extends import Extends

def ExactProblem(DecoratedProblem):
    
    if hasattr(DecoratedProblem, "ProblemDecorators"):
        assert hasattr(DecoratedProblem, "UndecoratedProblemClass")
        @Extends(DecoratedProblem.UndecoratedProblemClass, preserve_class_name=True)
        class ExactProblem_Class(DecoratedProblem.UndecoratedProblemClass):
            pass
            
        setattr(ExactProblem_Class, "__name__", "Exact" + ExactProblem_Class.__name__)
        
        # Re-apply decorators, replacing e.g. EIM with ExactParametrizedFunctions:
        for (Decorator, ExactDecorator, kwargs) in zip(DecoratedProblem.ProblemDecorators, DecoratedProblem.ProblemExactDecorators, DecoratedProblem.ProblemDecoratorsKwargs):
            if ExactDecorator is not None:
                ExactProblem_Class = ExactDecorator(**kwargs)(ExactProblem_Class)
            else:
                ExactProblem_Class = Decorator(**kwargs)(ExactProblem_Class)
                
        # Return
        return ExactProblem_Class
    else:
        return DecoratedProblem
        
