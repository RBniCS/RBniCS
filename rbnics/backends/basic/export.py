# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import overload
from rbnics.utils.io import Folders


def export(backend, wrapping):

    class _Export(object):
        @overload(backend.Function.Type(), (Folders.Folder, str), str, (int, None), None)
        def __call__(self, solution, directory, filename, suffix, component):
            wrapping.function_save(solution, directory, filename, suffix=suffix)

        @overload(backend.Function.Type(), (Folders.Folder, str), str, (int, None), (int, str))
        def __call__(self, solution, directory, filename, suffix, component):
            restricted_solution = wrapping.function_extend_or_restrict(
                solution, component, wrapping.get_function_subspace(solution, component), None, weight=None, copy=True)
            wrapping.function_save(restricted_solution, directory, filename, suffix=suffix)

        @overload((backend.Matrix.Type(), backend.Vector.Type()), (Folders.Folder, str), str, None, None)
        def __call__(self, solution, directory, filename, suffix, component):
            wrapping.tensor_save(solution, directory, filename)

    return _Export()
