# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


class DelayedProduct(object):
    def __init__(self, arg):
        assert not isinstance(arg, DelayedProduct)
        self._args = [arg]

    def __imul__(self, other):
        assert not isinstance(other, DelayedProduct)
        self._args.append(other)
        return self
