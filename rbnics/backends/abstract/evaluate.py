# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import abstract_backend


# Evaluate a parametrized expression, possibly at a specific location
@abstract_backend
def evaluate(expression, at=None, **kwargs):
    pass
