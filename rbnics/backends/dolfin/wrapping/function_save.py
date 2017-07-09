# Copyright (C) 2015-2017 by the RBniCS authors
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

from dolfin import assign, File
from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import function_extend_or_restrict
from rbnics.backends.dolfin.wrapping.get_function_subspace import get_function_subspace

def function_save(fun, directory, filename, suffix=None):
    fun_V = fun.function_space()
    if hasattr(fun_V, "_component_to_index") and len(fun_V._component_to_index) > 1:
        for (component, index) in fun_V._component_to_index.iteritems():
            sub_fun = function_extend_or_restrict(fun, component, get_function_subspace(fun_V, component), None, weight=None, copy=True)
            _write_to_pvd_file(sub_fun, directory, filename + "_component_" + str(index), suffix)
    else:
        _write_to_pvd_file(fun, directory, filename, suffix)
    if suffix is not None:
        filename = filename + "." + str(suffix)
    full_filename = str(directory) + "/" + filename + ".xml"
    file = File(full_filename)
    file << fun
    
def _write_to_pvd_file(fun, directory, filename, suffix):
    full_filename = str(directory) + "/" + filename + ".pvd"
    if suffix is not None:
        if full_filename in _all_pvd_files:
            assert _all_pvd_latest_suffix[full_filename] == suffix - 1
            _all_pvd_latest_suffix[full_filename] = suffix
        else:
            assert suffix == 0
            _all_pvd_files[full_filename] = File(full_filename, "compressed")
            _all_pvd_latest_suffix[full_filename] = 0
            _all_pvd_functions[full_filename] = fun.copy(deepcopy=True)
        # Make sure to always use the same function, otherwise dolfin
        # changes the numbering and visualization is difficult in ParaView
        assign(_all_pvd_functions[full_filename], fun)
        _all_pvd_files[full_filename] << _all_pvd_functions[full_filename]
    else:
        file = File(full_filename, "compressed")
        file << fun
    
_all_pvd_files = dict()
_all_pvd_latest_suffix = dict()
_all_pvd_functions = dict()
