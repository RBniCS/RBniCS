# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import inspect
import types
from rbnics.utils.cache import cache
from rbnics.utils.decorators.dispatch import dispatch


def ReducedProblemDecoratorFor(Algorithm, replaces=None, replaces_if=None, exact_decorator_for=None):
    # Convert replaces into a reduced problem decorator generator
    if replaces is not None:
        assert inspect.isfunction(replaces)
        replaces = _ReducedProblemDecoratorGenerator(replaces)

    # Prepare decorator
    def ReducedProblemDecoratorFor_Decorator(ReducedProblemDecorator):
        # Prepare a reduced problem decorator generator
        assert inspect.isfunction(ReducedProblemDecorator)
        ReducedProblemDecoratorGenerator = _ReducedProblemDecoratorGenerator(ReducedProblemDecorator)
        # Add to cache ((object, object) is a placeholder for (Problem, ReductionMethod) types)
        dispatch(*(object, object), name=Algorithm.__name__, module=_cache, replaces=replaces,
                 replaces_if=replaces_if)(ReducedProblemDecoratorGenerator)
        # Return unchanged reduced problem decorator
        return ReducedProblemDecorator

    return ReducedProblemDecoratorFor_Decorator


@cache
def _ReducedProblemDecoratorGenerator(ReducedProblemDecorator):

    def _ReducedProblemDecoratorGenerator_Function(truth_problem, reduction_method, **kwargs):
        return ReducedProblemDecorator

    return _ReducedProblemDecoratorGenerator_Function


_cache = types.ModuleType("reduced problem decorators", "Storage for reduced problem decorators")
