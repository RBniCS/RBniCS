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

import inspect
import types
from rbnics.utils.cache import cache
from rbnics.utils.decorators.dispatch import dispatch

def ReducedProblemFor(Problem, ReductionMethod, replaces=None, replaces_if=None):
    # Convert replaces into a reduced problem generator
    if replaces is not None:
        assert inspect.isclass(replaces)
        replaces = _ReducedProblemGenerator(replaces)
    # Prepare decorator
    def ReducedProblemFor_Decorator(ReducedProblem):
        # Prepare a reduced problem generator
        assert inspect.isclass(ReducedProblem)
        ReducedProblemGenerator = _ReducedProblemGenerator(ReducedProblem)
        # Add to cache.
        dispatch(*(Problem, ReductionMethod), name="ReducedProblem", module=_cache, replaces=replaces, replaces_if=replaces_if)(ReducedProblemGenerator)
        # Return unchanged reduced problem
        return ReducedProblem
    return ReducedProblemFor_Decorator

@cache
def _ReducedProblemGenerator(ReducedProblem):
    def _ReducedProblemGenerator_Function(truth_problem, reduction_method, **kwargs):
        return ReducedProblem
    return _ReducedProblemGenerator_Function
    
_cache = types.ModuleType("reduced problems", "Storage for reduced problems")
