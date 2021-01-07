# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.utils.decorators import dict_of, list_of, overload


def SnapshotsMatrix(FunctionsList):

    class _SnapshotsMatrix(FunctionsList):

        @overload(FunctionsList, (None, str, dict_of(str, str)), (None, list_of(Number)), bool)
        def _enrich(self, functions, component, weights, copy):
            if weights is not None:
                assert len(weights) == len(functions)
                for (index, function) in enumerate(functions):
                    self._add_to_list(function, component, weights[index], copy)
            else:
                for function in functions:
                    self._add_to_list(function, component, None, copy)

    return _SnapshotsMatrix
