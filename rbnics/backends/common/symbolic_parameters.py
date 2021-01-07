# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.abstract import SymbolicParameters as AbstractSymbolicParameters
from rbnics.utils.decorators import BackendFor, tuple_of


# Handle the trivial case of a non-parametric problem, that is mu = ()
@BackendFor("common", inputs=(object, object, tuple_of(())))
class SymbolicParameters(AbstractSymbolicParameters, tuple):
    def __new__(cls, problem, V, mu):
        return tuple.__new__(cls, ())

    def __str__(self):
        assert len(self) == 0
        return "()"
