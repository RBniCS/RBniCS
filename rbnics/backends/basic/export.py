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

def export(backend, wrapping):
    class _Export(object):
        @overload(backend.Function.Type(), (Folders.Folder, str), str, (int, None), None)
        def __call__(self, solution, directory, filename, suffix, component):
            wrapping.function_save(solution, directory, filename, suffix=suffix)
        
        @overload(backend.Function.Type(), (Folders.Folder, str), str, (int, None), (int, str))
        def __call__(self, solution, directory, filename, suffix, component):
            restricted_solution = wrapping.function_extend_or_restrict(solution, component, wrapping.get_function_subspace(solution, component), None, weight=None, copy=True)
            wrapping.function_save(restricted_solution, directory, filename, suffix=suffix)
        
        @overload((backend.Matrix.Type(), backend.Vector.Type()), (Folders.Folder, str), str, None, None)
        def __call__(self, solution, directory, filename, suffix, component):
            wrapping.tensor_save(solution, directory, filename)
    return _Export()
