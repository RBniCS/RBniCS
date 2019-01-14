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

from rbnics.backends.basic import import_ as basic_import_
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import build_dof_map_reader_mapping, form_argument_space, function_extend_or_restrict, function_load, get_function_space, get_function_subspace, to_petsc4py
from rbnics.backends.dolfin.wrapping.tensor_load import basic_tensor_load
from rbnics.utils.decorators import backend_for, ModuleWrapper
from rbnics.utils.io import Folders

backend = ModuleWrapper(Function, Matrix, Vector)
wrapping_for_wrapping = ModuleWrapper(build_dof_map_reader_mapping, form_argument_space, to_petsc4py)
tensor_load = basic_tensor_load(backend, wrapping_for_wrapping)
wrapping = ModuleWrapper(function_extend_or_restrict, function_load, get_function_space, get_function_subspace, tensor_load=tensor_load)
import_base = basic_import_(backend, wrapping)

# Import a solution from file
@backend_for("dolfin", inputs=((Function.Type(), Matrix.Type(), Vector.Type()), (Folders.Folder, str), str, (int, None), (int, str, None)))
def import_(solution, directory, filename, suffix=None, component=None):
    import_base(solution, directory, filename, suffix, component)
