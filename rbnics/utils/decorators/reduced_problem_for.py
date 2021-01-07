# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
        dispatch(*(Problem, ReductionMethod), name="ReducedProblem", module=_cache, replaces=replaces,
                 replaces_if=replaces_if)(ReducedProblemGenerator)
        # Return unchanged reduced problem
        return ReducedProblem

    return ReducedProblemFor_Decorator


@cache
def _ReducedProblemGenerator(ReducedProblem):

    def _ReducedProblemGenerator_Function(truth_problem, reduction_method, **kwargs):
        return ReducedProblem

    return _ReducedProblemGenerator_Function


_cache = types.ModuleType("reduced problems", "Storage for reduced problems")
