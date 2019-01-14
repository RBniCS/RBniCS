# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

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
            restricted_solution = wrapping.function_extend_or_restrict(solution, component, subspace, None, weight=None, copy=True)
            wrapping.function_load(restricted_solution, directory, filename, suffix=suffix)
            wrapping.function_extend_or_restrict(restricted_solution, None, space, component, weight=None, copy=True, extended_or_restricted_function=solution)
        
        @overload((backend.Matrix.Type(), backend.Vector.Type()), (Folders.Folder, str), str, None, None)
        def __call__(self, solution, directory, filename, suffix, component):
            wrapping.tensor_load(solution, directory, filename)
    return _Import()
