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

def ReductionMethodFor(Problem, category, replaces=None, replaces_if=None):
    # Convert replaces into a reduction method generator
    if replaces is not None:
        assert inspect.isclass(replaces)
        replaces = _ReductionMethodGenerator(replaces)
    # Prepare decorator
    def ReductionMethodFor_Decorator(ReductionMethod):
        # Prepare a reduction method generator
        assert inspect.isclass(ReductionMethod)
        ReductionMethodGenerator = _ReductionMethodGenerator(ReductionMethod)
        # Add to cache
        dispatch(Problem, name=category, module=_cache, replaces=replaces, replaces_if=replaces_if)(ReductionMethodGenerator)
        # Return unchanged reduction method
        return ReductionMethod
    return ReductionMethodFor_Decorator

@cache
def _ReductionMethodGenerator(ReductionMethod):
    def _ReductionMethodGenerator_Function(truth_problem, **kwargs):
        return ReductionMethod
    return _ReductionMethodGenerator_Function
    
_cache = types.ModuleType("reduction methods", "Storage for reduction methods")
