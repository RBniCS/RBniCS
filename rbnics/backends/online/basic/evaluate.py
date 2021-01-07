# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import overload, tuple_of


def evaluate(backend):

    class _Evaluate(object):
        @overload(backend.Matrix.Type(), None)
        def __call__(self, matrix, at):
            return matrix

        @overload(backend.Matrix.Type(), tuple_of(int))
        def __call__(self, matrix, at):
            assert len(at) == 2
            return matrix[at]

        @overload(backend.Vector.Type(), None)
        def __call__(self, vector, at):
            return vector

        @overload(backend.Vector.Type(), tuple_of(int))
        def __call__(self, vector, at):
            assert len(at) == 1
            return vector

    return _Evaluate()
