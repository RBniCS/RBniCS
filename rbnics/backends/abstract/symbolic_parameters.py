# Copyright (C) 2015-2022 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ABCMeta, AbstractBackend


@AbstractBackend
class SymbolicParameters(object, metaclass=ABCMeta):
    def __init__(cls, problem, space, mu):
        pass
