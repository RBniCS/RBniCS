# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.basic.wrapping.delayed_product import DelayedProduct


class DelayedSum(object):
    def __init__(self, arg):
        assert not isinstance(arg, DelayedSum)
        self._args = [arg]

    def __iadd__(self, other):
        assert isinstance(other, DelayedProduct)
        self._args.append(other)
        return self
