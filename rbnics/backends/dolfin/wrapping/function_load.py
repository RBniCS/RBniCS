# Copyright (C) 2015-2018 by the RBniCS authors
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

import os
from ufl import product
from dolfin import Function
from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import function_extend_or_restrict
from rbnics.backends.dolfin.wrapping.get_function_subspace import get_function_subspace
from rbnics.backends.dolfin.wrapping.function_save import _all_solution_files
from rbnics.backends.dolfin.wrapping.function_save import SolutionFile
from rbnics.utils.mpi import is_io_process
from rbnics.utils.io import TextIO as SuffixIO

def function_load(fun, directory, filename, suffix=None):
    fun_V = fun.function_space()
    if hasattr(fun_V, "_index_to_components") and len(fun_V._index_to_components) > 1:
        fun.vector().zero()
        for (index, components) in fun_V._index_to_components.items():
            sub_fun_V = get_function_subspace(fun_V, components)
            sub_fun = Function(sub_fun_V)
            if not _read_from_file(sub_fun, directory, filename, suffix, components):
                return False
            else:
                extended_sub_fun = function_extend_or_restrict(sub_fun, None, fun_V, components[0], weight=None, copy=True)
                fun.vector().add_local(extended_sub_fun.vector().get_local())
                fun.vector().apply("add")
        return True
    else:
        return _read_from_file(fun, directory, filename, suffix)
    
def _read_from_file(fun, directory, filename, suffix, components=None):
    if components is not None:
        filename = filename + "_component_" + "".join(components)
        function_name = "function_" + "".join(components)
    else:
        function_name = "function"
    fun_rank = fun.value_rank()
    fun_dim = product(fun.value_shape())
    assert fun_rank <= 2
    if (
        (fun_rank is 1 and fun_dim not in (2, 3))
            or
        (fun_rank is 2 and fun_dim not in (4, 9))
    ):
        funs = fun.split(deepcopy=True)
        for (i, fun_i) in enumerate(funs):
            if components is not None:
                filename_i = filename + "_subcomponent_" + str(i)
            else:
                filename_i = filename + "_component_" + str(i)
            if not _read_from_file(fun_i, directory, filename_i, suffix, None):
                return False
            else:
                assign(fun.sub(i), fun_i)
        return True
    else:
        if suffix is not None:
            if suffix is 0:
                # Remove from storage and re-create
                _all_solution_files.pop((directory, filename), None)
                _all_solution_files[(directory, filename)] = SolutionFile(directory, filename)
            file_ = _all_solution_files[(directory, filename)]
            return file_.read(fun, function_name, suffix)
        else:
            file_ = SolutionFile(directory, filename)
            return file_.read(fun, function_name, 0)
