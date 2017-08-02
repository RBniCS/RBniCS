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

import os # for path
from dolfin import File, Function, has_hdf5, has_hdf5_parallel, XDMFFile
from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import function_extend_or_restrict
from rbnics.utils.mpi import is_io_process

def has_hdf5():
    return False # Temporarily disable output to XDMFFile until next FEniCS release

def function_load(fun, directory, filename, suffix=None):
    if not has_hdf5() or not has_hdf5_parallel():
        return _read_from_xml_file(fun, directory, filename, suffix)
    else:
        fun_V = fun.function_space()
        if hasattr(fun_V, "_component_to_index") and len(fun_V._component_to_index) > 1:
            for (component, index) in fun_V._component_to_index.iteritems():
                sub_fun_V = get_function_subspace(fun_V, component)
                sub_fun = Function(sub_fun_V)
                if not _read_from_xdmf_file(sub_fun, directory, filename, suffix, component, component):
                    return False
                else:
                    extended_sub_fun = function_extend_or_restrict(sub_fun, None, fun_V, component, weight=None, copy=True)
                    fun.vector().add_local(extended_sub_fun.vector().array())
                    fun.vector().apply("add")
            return True
        else:
            return _read_from_xdmf_file(fun, directory, filename, suffix)
    
def _read_from_xml_file(fun, directory, filename, suffix):
    if suffix is not None:
        filename = filename + "." + str(suffix)
    full_filename = str(directory) + "/" + filename + ".xml"
    file_exists = False
    if is_io_process() and os.path.exists(full_filename):
        file_exists = True
    file_exists = is_io_process.mpi_comm.bcast(file_exists, root=is_io_process.root)
    if file_exists:
        file_ = File(full_filename)
        file_ >> fun
    return file_exists
    
def _read_from_xdmf_file(fun, directory, filename, suffix, component=None, function_name=None):
    if component is not None:
        filename = filename + "_component_" + component
    if function_name is None:
        function_name = "function"
    fun_rank = fun.value_rank()
    fun_dim = fun.value_size()
    assert fun_rank <= 2
    if (
        (fun_rank is 1 and fun_dim not in (2, 3))
            or
        (fun_rank is 2 and fun_dim not in (4, 9))
    ):
        fun_V = fun.function_space()
        for i in range(fun_dim):
            if component is not None:
                filename_i = filename + "_subcomponent_" + str(i)
            else:
                filename_i = filename + "_component_" + str(i)
            fun_i_V = get_function_subspace(fun_V, i)
            fun_i = Function(fun_i_V)
            if not _read_from_xdmf_file(fun_i, directory, filename, suffix, None, function_name):
                return False
            else:
                extended_sub_fun = function_extend_or_restrict(fun_i, None, fun_V, component, weight=None, copy=True)
                fun.vector().add_local(extended_sub_fun.vector().array())
                fun.vector().apply("add")
    else:
        full_filename_checkpoint = str(directory) + "/" + filename + "_checkpoint.xdmf"
        file_exists = False
        if is_io_process() and os.path.exists(full_filename_checkpoint):
            file_exists = True
        file_exists = is_io_process.mpi_comm.bcast(file_exists, root=is_io_process.root)
        if file_exists:
            if suffix is not None:
                if full_filename_checkpoint in _all_xdmf_files:
                    assert _all_xdmf_latest_suffix[full_filename_checkpoint] == suffix - 1
                    _all_xdmf_latest_suffix[full_filename_checkpoint] = suffix
                else:
                    assert suffix == 0
                    _all_xdmf_files[full_filename_checkpoint] = XDMFFile(full_filename_checkpoint)
                    _all_xdmf_latest_suffix[full_filename_checkpoint] = 0
                _all_xdmf_files[full_filename_checkpoint].read_checkpoint(fun, function_name, suffix)
            else:
                file_checkpoint = XDMFFile(full_filename_checkpoint)
                file_checkpoint.read_checkpoint(fun, function_name, 0)
                file_checkpoint.close()
        return file_exists
    
_all_xdmf_files = dict()
_all_xdmf_latest_suffix = dict()
