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
from dolfin import assign, File, has_hdf5, has_hdf5_parallel, XDMFFile
from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import function_extend_or_restrict
from rbnics.backends.dolfin.wrapping.get_function_subspace import get_function_subspace
from rbnics.utils.mpi import is_io_process

def has_hdf5():
    return False # Temporarily disable output to XDMFFile until next FEniCS release

def function_save(fun, directory, filename, suffix=None):
    fun_V = fun.function_space()
    if not has_hdf5() or not has_hdf5_parallel():
        if hasattr(fun_V, "_component_to_index") and len(fun_V._component_to_index) > 1:
            for (component, index) in fun_V._component_to_index.items():
                sub_fun = function_extend_or_restrict(fun, component, get_function_subspace(fun_V, component), None, weight=None, copy=True)
                _write_to_pvd_file(sub_fun, directory, filename, suffix, component)
        else:
            _write_to_pvd_file(fun, directory, filename, suffix)
        _write_to_xml_file(fun, directory, filename, suffix)
    else:
        if hasattr(fun_V, "_component_to_index") and len(fun_V._component_to_index) > 1:
            for (component, index) in fun_V._component_to_index.items():
                sub_fun = function_extend_or_restrict(fun, component, get_function_subspace(fun_V, component), None, weight=None, copy=True)
                _write_to_xdmf_file(sub_fun, directory, filename, suffix, component, component)
        else:
            _write_to_xdmf_file(fun, directory, filename, suffix)
    
def _write_to_xml_file(fun, directory, filename, suffix):
    if suffix is not None:
        filename = filename + "." + str(suffix)
    full_filename = str(directory) + "/" + filename + ".xml"
    file_ = File(full_filename)
    file_ << fun
    
def _write_to_pvd_file(fun, directory, filename, suffix, component=None):
    if component is not None:
        filename = filename + "_component_" + component
    fun_rank = fun.value_rank()
    fun_dim = fun.value_size()
    assert fun_rank <= 2
    if (
        (fun_rank is 1 and fun_dim not in (2, 3))
            or
        (fun_rank is 2 and fun_dim not in (4, 9))
    ):
        funs = fun.split(deepcopy=True)
        for (i, fun_i) in enumerate(funs):
            if component is not None:
                filename_i = filename + "_subcomponent_" + str(i)
            else:
                filename_i = filename + "_component_" + str(i)
            _write_to_pvd_file(fun_i, directory, filename_i, suffix)
    else:
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
            file_ = File(full_filename, "compressed")
            file_ << fun
    
def _write_to_xdmf_file(fun, directory, filename, suffix, component=None, function_name=None):
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
        funs = fun.split(deepcopy=True)
        for (i, fun_i) in enumerate(funs):
            if component is not None:
                filename_i = filename + "_subcomponent_" + str(i)
            else:
                filename_i = filename + "_component_" + str(i)
            _write_to_xdmf_file(fun_i, directory, filename_i, suffix, None, function_name)
    else:
        full_filename_visualization = str(directory) + "/" + filename + ".xdmf"
        full_filename_checkpoint = str(directory) + "/" + filename + "_checkpoint.xdmf"
        if suffix is not None:
            if full_filename_checkpoint in _all_xdmf_files:
                assert _all_xdmf_latest_suffix[full_filename_checkpoint] == suffix - 1
                _all_xdmf_latest_suffix[full_filename_checkpoint] = suffix
            else:
                assert suffix == 0
                # Remove existing files if any, as new functions should not be appended,
                # but rather overwrite existing functions
                if is_io_process() and os.path.exists(full_filename_checkpoint):
                    os.remove(full_filename_checkpoint)
                    os.remove(full_filename_checkpoint.replace(".xdmf", ".h5"))
                _all_xdmf_files[full_filename_visualization] = XDMFFile(full_filename_visualization)
                _all_xdmf_files[full_filename_checkpoint] = XDMFFile(full_filename_checkpoint)
                _all_xdmf_latest_suffix[full_filename_checkpoint] = 0                     # don't store these twice for both visualization
                _all_xdmf_functions[full_filename_checkpoint] = fun.copy(deepcopy=True)   # and checkpoint, as they are the same!
            # Make sure to always use the same function, otherwise dolfin
            # changes the numbering and visualization is difficult in ParaView
            assign(_all_xdmf_functions[full_filename_checkpoint], fun)
            _all_xdmf_files[full_filename_visualization].write(_all_xdmf_functions[full_filename_checkpoint], suffix)
            _all_xdmf_files[full_filename_checkpoint].write_checkpoint(_all_xdmf_functions[full_filename_checkpoint], function_name, suffix)
        else:
            # Remove existing files if any, as new functions should not be appended,
            # but rather overwrite existing functions
            if is_io_process() and os.path.exists(full_filename_checkpoint):
                os.remove(full_filename_checkpoint)
                os.remove(full_filename_checkpoint.replace(".xdmf", ".h5"))
            file_visualization = XDMFFile(full_filename_visualization)
            file_visualization.write(fun, 0)
            file_visualization.close()
            file_checkpoint = XDMFFile(full_filename_checkpoint)
            file_checkpoint.write_checkpoint(fun, function_name, 0)
            file_checkpoint.close()
        
_all_pvd_files = dict()
_all_pvd_latest_suffix = dict()
_all_pvd_functions = dict()

_all_xdmf_files = dict()
_all_xdmf_latest_suffix = dict()
_all_xdmf_functions = dict()
