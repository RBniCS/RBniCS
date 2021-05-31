# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from ufl import MixedElement, TensorElement, VectorElement
from dolfin import assign, File as PVDFile, File as XMLFile, has_hdf5, has_hdf5_parallel, XDMFFile
from rbnics.utils.cache import Cache
from rbnics.utils.io import TextIO as IndexIO
from rbnics.utils.mpi import parallel_io


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

        def remove_files_task():
            if os.path.exists(full_filename + "_index.sfx"):
                os.remove(full_filename + "_index.sfx")

        parallel_io(remove_files_task)

    def write(self, function, name, index):
        pass

    def read(self, function, name, index):
        pass

    def _update_function_container(self, function):
        if self._function_container is None:
            self._function_container = function.copy(deepcopy=True)
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


class SolutionFileXML(SolutionFile_Base):
    def __init__(self, directory, filename):
        SolutionFile_Base.__init__(self, directory, filename)
        self._visualization_file = PVDFile(self._full_filename + ".pvd", "compressed")

    @staticmethod
    def remove_files(directory, filename):
        SolutionFile_Base.remove_files(directory, filename)
        # No need to remove further files, PVD and XML will get automatically truncated

    def write(self, function, name, index):
        assert index in (self._last_index, self._last_index + 1)
        if index == self._last_index + 1:  # writing out solutions after time stepping
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


class SolutionFileXDMF(SolutionFile_Base):
    # DOLFIN 2018.1.0.dev added (throughout the developement cycle) an optional append
    # attribute to XDMFFile.write_checkpoint, which should be set to its non-default value,
    # thus breaking backwards compatibility
    append_attribute = XDMFFile.write_checkpoint.__doc__.find("append: bool") > - 1

    def __init__(self, directory, filename):
        SolutionFile_Base.__init__(self, directory, filename)
        self._visualization_file = XDMFFile(self._full_filename + ".xdmf")
        self._visualization_file.parameters["flush_output"] = True
        self._restart_filename = self._full_filename + "_checkpoint.xdmf"
        self._restart_file = XDMFFile(self._restart_filename)
        self._restart_file.parameters["flush_output"] = True

    @staticmethod
    def remove_files(directory, filename):
        SolutionFile_Base.remove_files(directory, filename)
        #
        full_filename = os.path.join(str(directory), filename)

        def remove_files_task():
            if os.path.exists(full_filename + ".xdmf"):
                os.remove(full_filename + ".xdmf")
                os.remove(full_filename + ".h5")
                os.remove(full_filename + "_checkpoint.xdmf")
                os.remove(full_filename + "_checkpoint.h5")

        parallel_io(remove_files_task)

    def write(self, function, name, index):
        time = float(index)
        # Write visualization file (no append available, will overwrite)
        self._update_function_container(function)
        self._visualization_file.write(self._function_container, time)
        # Write restart file. It might be possible that the solution was written to file in a previous run
        # and the execution was interrupted before last written index was updated. In this corner case
        # there would be two functions corresponding to the same time, with two consecutive indices.
        # For now the inelegant way is to try to read: if that works, assume that we are in the corner case;
        # otherwise, we are in the standard case and we should write to file.
        try:
            if os.path.exists(self._restart_filename):
                self._restart_file.read_checkpoint(self._function_container, name, index)
            else:
                raise RuntimeError
        except RuntimeError:
            from dolfin.cpp.log import get_log_level, LogLevel, set_log_level
            self._update_function_container(function)
            bak_log_level = get_log_level()
            set_log_level(int(LogLevel.WARNING) + 1)  # disable xdmf logs
            if self.append_attribute:
                self._restart_file.write_checkpoint(self._function_container, name, time, append=True)
            else:
                self._restart_file.write_checkpoint(self._function_container, name, time)
            set_log_level(bak_log_level)
            # Once solutions have been written to file, update last written index
        self._write_last_index(index)

    def read(self, function, name, index):
        if index <= self._last_index:
            time = float(index)
            assert os.path.exists(self._restart_filename)
            self._restart_file.read_checkpoint(function, name, index)
            self._update_function_container(function)
            self._visualization_file.write(self._function_container, time)  # because no append option is available
        else:
            raise OSError


def function_save(fun, directory, filename, suffix=None):
    fun_V = fun.function_space()
    if hasattr(fun_V, "_index_to_components") and len(fun_V._index_to_components) > 1:
        for (index, components) in fun_V._index_to_components.items():
            _write_to_file(fun.sub(index, deepcopy=True), directory, filename, suffix, components)
    else:
        _write_to_file(fun, directory, filename, suffix)


def _write_to_file(fun, directory, filename, suffix, components=None):
    if components is not None:
        filename = filename + "_component_" + "".join(components)
        function_name = "function_" + "".join(components)
    else:
        function_name = "function"
    fun_V_element = fun.function_space().ufl_element()
    if isinstance(fun_V_element, MixedElement) and not isinstance(fun_V_element, (TensorElement, VectorElement)):
        funs = fun.split(deepcopy=True)
        for (i, fun_i) in enumerate(funs):
            if components is not None:
                filename_i = filename + "_subcomponent_" + str(i)
            else:
                filename_i = filename + "_component_" + str(i)
            _write_to_file(fun_i, directory, filename_i, suffix, None)
    else:
        if fun_V_element.family() == "Real":
            SolutionFile = SolutionFileXML
        else:
            if has_hdf5() and has_hdf5_parallel():
                SolutionFile = SolutionFileXDMF
            else:
                SolutionFile = SolutionFileXML
        if suffix is not None:
            if suffix == 0:
                # Remove existing files if any, as new functions should not be appended,
                # but rather overwrite existing functions
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
            # Remove existing files if any, as new functions should not be appended,
            # but rather overwrite existing functions
            SolutionFile.remove_files(directory, filename)
            # Write function to file
            file_ = SolutionFile(directory, filename)
            file_.write(fun, function_name, 0)


_all_solution_files = Cache()
