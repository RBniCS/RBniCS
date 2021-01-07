# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import overload
from rbnics.utils.io import Folders


# Returns True if it was possible to import the file, False otherwise
def import_(backend, wrapping):

    class _Import(object):
        @overload(backend.Function.Type(), (Folders.Folder, str), str, (int, None), None)
        def __call__(self, solution, directory, filename, suffix, component):
            wrapping.function_load(solution, directory, filename, suffix=suffix)

        @overload(backend.Function.Type(), (Folders.Folder, str), str, (int, None), (int, str))
        def __call__(self, solution, directory, filename, suffix, component):
            space = wrapping.get_function_space(solution)
            subspace = wrapping.get_function_subspace(solution, component)
            restricted_solution = wrapping.function_extend_or_restrict(solution, component, subspace, None,
                                                                       weight=None, copy=True)
            wrapping.function_load(restricted_solution, directory, filename, suffix=suffix)
            wrapping.function_extend_or_restrict(restricted_solution, None, space, component, weight=None,
                                                 copy=True, extended_or_restricted_function=solution)

        @overload((backend.Matrix.Type(), backend.Vector.Type()), (Folders.Folder, str), str, None, None)
        def __call__(self, solution, directory, filename, suffix, component):
            wrapping.tensor_load(solution, directory, filename)

    return _Import()
