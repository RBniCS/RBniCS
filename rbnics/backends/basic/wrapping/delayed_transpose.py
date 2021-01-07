# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


class DelayedTranspose(object):
    def __init__(self, arg):
        self._args = [arg]

    def __mul__(self, other):
        self._args.append(other)
        return self
