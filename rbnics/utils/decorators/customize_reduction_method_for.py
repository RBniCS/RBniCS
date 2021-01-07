# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import inspect
from rbnics.utils.cache import Cache
from rbnics.utils.jupyter import is_jupyter


def CustomizeReductionMethodFor(Problem):
    assert inspect.isabstract(Problem), (
        "It is suggested to use this customizer for abstract classes (e.g., before specifying theta terms"
        + " and operators, or decorating with EIM or SCM), because otherwise the customization would not"
        + " be preserved with a call to exact_problem.")

    def CustomizeReductionMethodFor_Decorator(customizer):
        if not is_jupyter():
            assert Problem not in _cache
        _cache[Problem] = customizer
        return customizer

    return CustomizeReductionMethodFor_Decorator


_cache = Cache()  # from Problem to decorator
