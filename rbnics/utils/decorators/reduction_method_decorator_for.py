# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import inspect
import types
from rbnics.utils.cache import cache
from rbnics.utils.decorators.dispatch import dispatch


def ReductionMethodDecoratorFor(Algorithm, replaces=None, replaces_if=None, exact_decorator_for=None):
    # Convert replaces into a reduction method decorator generator
    if replaces is not None:
        assert inspect.isfunction(replaces)
        replaces = _ReductionMethodDecoratorGenerator(replaces)

    # Prepare decorator
    def ReductionMethodDecoratorFor_Decorator(ReductionMethodDecorator):
        # Prepare a reduction method decorator generator
        assert inspect.isfunction(ReductionMethodDecorator)
        ReductionMethodDecoratorGenerator = _ReductionMethodDecoratorGenerator(ReductionMethodDecorator)
        # Add to cache (object is a placeholder for Problem type)
        dispatch(object, name=Algorithm.__name__, module=_cache, replaces=replaces,
                 replaces_if=replaces_if)(ReductionMethodDecoratorGenerator)
        # Return unchanged reduction method decorator
        return ReductionMethodDecorator

    return ReductionMethodDecoratorFor_Decorator


@cache
def _ReductionMethodDecoratorGenerator(ReductionMethodDecorator):
    def _ReductionMethodDecoratorGenerator_Function(truth_problem, **kwargs):
        return ReductionMethodDecorator
    return _ReductionMethodDecoratorGenerator_Function


_cache = types.ModuleType("reduction method decorators", "Storage for reduction method decorators")
