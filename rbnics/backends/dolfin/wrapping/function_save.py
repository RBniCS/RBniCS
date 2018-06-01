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
from dolfin import assign, File as PVDFile, File as XMLFile, has_hdf5, has_hdf5_parallel, has_pybind11, XDMFFile
if has_pybind11():
    from dolfin.cpp.log import get_log_level, LogLevel, set_log_level
    WARNING = LogLevel.WARNING
else:
    from dolfin import get_log_level, set_log_level, WARNING
from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import function_extend_or_restrict
from rbnics.backends.dolfin.wrapping.get_function_subspace import get_function_subspace
from rbnics.utils.cache import Cache
from rbnics.utils.io import TextIO as IndexIO
from rbnics.utils.mpi import is_io_process

class SolutionFile_Base(object):
    def __init__(self, directory, filename):
        self._directory = directory
        self._filename = filename
        self._full_filename = os.path.join(str(directory), filename)
        self._last_index = None
        self._function_container = None
        self._init_last_index()
        
    @staticmethod
    def remove_files(directory, filename):
        full_filename = os.path.join(str(directory), filename)
        if is_io_process() and os.path.exists(full_filename + "_index.sfx"):
            os.remove(full_filename + "_index.sfx")
        
    def write(self, function, name, index):
        pass
        
    def read(self, function, name, index):
        pass
        
    def _update_function_container(self, function):
        if self._function_container is None:
            self._function_container = function.copy()
        else:
            assign(self._function_container, function)
        
    def _init_last_index(self):
        if IndexIO.exists_file(self._directory, self._filename + "_index.sfx"):
            self._last_index = IndexIO.load_file(self._directory, self._filename + "_index.sfx")
        else:
            self._last_index = -1
            
    def _write_last_index(self, index):
        self._last_index = index
        # Write out current index
        IndexIO.save_file(index, self._directory, self._filename + "_index.sfx")

if not has_hdf5() or not has_hdf5_parallel():
    class SolutionFile(SolutionFile_Base):
        def __init__(self, directory, filename):
            SolutionFile_Base.__init__(self, directory, filename)
            self._visualization_file = PVDFile(self._full_filename + ".pvd", "compressed")
            
        @staticmethod
        def remove_files(directory, filename):
            SolutionFile_Base.remove_files(directory, filename)
            # No need to remove further files, PVD and XML will get automatically truncated
            
        def write(self, function, name, index):
            assert index in (self._last_index, self._last_index + 1)
            if index == self._last_index + 1: # writing out solutions after time stepping
                self._update_function_container(function)
                self._visualization_file << self._function_container
                restart_file = XMLFile(self._full_filename + "_" + str(index) + ".xml")
                restart_file << self._function_container
                # Once solutions have been written to file, update last written index
                self._write_last_index(index)
            elif index == self._last_index:
                # corner case for problems with two (or more) unknowns which are written separately to file;
                # one unknown was written to file, while the other was not: since the problem might be coupled,
                # a recomputation of both is required, but there is no need to update storage
                pass
            else:
                raise ValueError("Invalid index")
            
        def read(self, function, name, index):
            if index <= self._last_index:
                restart_file = XMLFile(self._full_filename + "_" + str(index) + ".xml")
                restart_file >> function
            else:
                raise OSError
else:
    class SolutionFile(SolutionFile_Base):
        # DOLFIN 2018.1.0.dev added (throughout the developement cycle) an optional append
        # attribute to XDMFFile.write_checkpoint, which should be set to its non-default value,
        # thus breaking backwards compatibility
        append_attribute = XDMFFile.write_checkpoint.__doc__.find("append: bool") > - 1
        
        def __init__(self, directory, filename):
            SolutionFile_Base.__init__(self, directory, filename)
            self._visualization_file = XDMFFile(self._full_filename + ".xdmf")
            self._visualization_file.parameters["flush_output"] = True
            self._restart_file = XDMFFile(self._full_filename + "_checkpoint.xdmf")
            self._restart_file.parameters["flush_output"] = True
            
        @staticmethod
        def remove_files(directory, filename):
            SolutionFile_Base.remove_files(directory, filename)
            #
            full_filename = os.path.join(str(directory), filename)
            if is_io_process() and os.path.exists(full_filename + ".xdmf"):
                os.remove(full_filename + ".xdmf")
                os.remove(full_filename + ".h5")
                os.remove(full_filename + "_checkpoint.xdmf")
                os.remove(full_filename + "_checkpoint.h5")
            
        def write(self, function, name, index):
            assert index in (self._last_index, self._last_index + 1)
            if index == self._last_index + 1: # writing out solutions after time stepping
                self._update_function_container(function)
                time = float(index)
                self._visualization_file.write(self._function_container, time)
                bak_log_level = get_log_level()
                set_log_level(int(WARNING) + 1) # disable xdmf logs)
                if self.append_attribute:
                    self._restart_file.write_checkpoint(self._function_container, name, time, append=True)
                else:
                    self._restart_file.write_checkpoint(self._function_container, name, time)
                set_log_level(bak_log_level)
                # Once solutions have been written to file, update last written index
                self._write_last_index(index)
            elif index == self._last_index:
                # corner case for problems with two (or more) unknowns which are written separately to file;
                # one unknown was written to file, while the other was not: since the problem might be coupled,
                # a recomputation of both is required, but there is no need to update storage
                pass
            else:
                raise ValueError("Invalid index")
            
        def read(self, function, name, index):
            if index <= self._last_index:
                time = float(index)
                self._restart_file.read_checkpoint(function, name, index)
                self._update_function_container(function)
                self._visualization_file.write(self._function_container, time)
            else:
                raise OSError

def function_save(fun, directory, filename, suffix=None):
    fun_V = fun.function_space()
    if hasattr(fun_V, "_index_to_components") and len(fun_V._index_to_components) > 1:
        for (index, components) in fun_V._index_to_components.items():
            sub_fun = function_extend_or_restrict(fun, components[0], get_function_subspace(fun_V, components[0]), None, weight=None, copy=True)
            _write_to_file(sub_fun, directory, filename, suffix, components)
    else:
        _write_to_file(fun, directory, filename, suffix)
    
def _write_to_file(fun, directory, filename, suffix, components=None):
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
            _write_to_file(fun_i, directory, filename_i, suffix, None)
    else:
        if suffix is not None:
            if suffix is 0:
                # Remove existing files if any, as new functions should not be appended, but rather overwrite existing functions
                SolutionFile.remove_files(directory, filename)
                # Remove from storage and re-create
                try:
                    del _all_solution_files[(directory, filename)]
                except KeyError:
                    pass
                _all_solution_files[(directory, filename)] = SolutionFile(directory, filename)
            file_ = _all_solution_files[(directory, filename)]
            file_.write(fun, function_name, suffix)
        else:
            # Remove existing files if any, as new functions should not be appended, but rather overwrite existing functions
            SolutionFile.remove_files(directory, filename)
            # Write function to file
            file_ = SolutionFile(directory, filename)
            file_.write(fun, function_name, 0)
        
_all_solution_files = Cache()
