# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import list_of, overload


def copy(backend, wrapping):

    class _Copy(object):
        @overload(backend.Function.Type(), )
        def __call__(self, arg):
            return wrapping.function_copy(arg)

        @overload(list_of(backend.Function.Type()), )
        def __call__(self, arg):
            output = list()
            for fun in arg:
                output.append(wrapping.function_copy(fun))
            return output

        @overload((backend.Matrix.Type(), backend.Vector.Type()), )
        def __call__(self, arg):
            return wrapping.tensor_copy(arg)

    return _Copy()
