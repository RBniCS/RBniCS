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

import os
from ufl import Form
from dolfin import as_backend_type
from petsc4py import PETSc
from rbnics.utils.mpi import is_io_process
from rbnics.utils.io import Folders, PickleIO
from rbnics.utils.decorators import overload

def basic_tensor_load(backend, wrapping):
    def _basic_tensor_load(tensor, directory, filename):
        mpi_comm = tensor.mpi_comm().tompi4py()
        form = tensor.generator._form
        load_failed = False
        # Read in generator
        full_filename_generator = os.path.join(str(directory), filename + ".generator")
        generator_string = None
        if is_io_process(mpi_comm):
            if os.path.exists(full_filename_generator):
                with open(full_filename_generator, "r") as generator_file:
                    generator_string = generator_file.readline()
            else:
                load_failed = True
        if mpi_comm.bcast(load_failed, root=is_io_process.root):
            return False
        else:
            generator_string = mpi_comm.bcast(generator_string, root=is_io_process.root)
        # Read in generator mpi size
        full_filename_generator_mpi_size = os.path.join(str(directory), filename + ".generator_mpi_size")
        generator_mpi_size_string = None
        if is_io_process(mpi_comm):
            if os.path.exists(full_filename_generator_mpi_size):
                with open(full_filename_generator_mpi_size, "r") as generator_mpi_size_file:
                    generator_mpi_size_string = generator_mpi_size_file.readline()
            else:
                load_failed = True
        if mpi_comm.bcast(load_failed, root=is_io_process.root):
            return False
        else:
            generator_mpi_size_string = mpi_comm.bcast(generator_mpi_size_string, root=is_io_process.root)
        # Read in generator mapping from processor dependent indices (at the time of saving) to processor independent (global_cell_index, cell_dof) tuple
        (permutation, loaded) = _permutation_load(tensor, directory, filename, form, generator_string + "_" + generator_mpi_size_string, mpi_comm)
        if not loaded:
            return False
        else:
            # Read in content
            return _tensor_load(tensor, directory, filename, permutation)
        
    @overload(backend.Matrix.Type(), (Folders.Folder, str), str, Form, str, object)
    def _permutation_load(tensor, directory, filename, form, form_name, mpi_comm):
        if form_name not in _permutation_storage:
            if not PickleIO.exists_file(directory, "." + form_name):
                return (None, False)
            else:
                V_0 = wrapping.form_argument_space(form, 0)
                V_1 = wrapping.form_argument_space(form, 1)
                V_0__dof_map_reader_mapping = wrapping.build_dof_map_reader_mapping(V_0)
                V_1__dof_map_reader_mapping = wrapping.build_dof_map_reader_mapping(V_1)
                (V_0__dof_map_writer_mapping, V_1__dof_map_writer_mapping) = PickleIO.load_file(directory, "." + form_name)
                matrix_row_permutation = dict() # from row index at time of saving to current row index
                matrix_col_permutation = dict() # from col index at time of saving to current col index
                (writer_mat, loaded) = _matrix_load(directory, filename)
                if not loaded:
                    return (None, False)
                writer_row_start, writer_row_end = writer_mat.getOwnershipRange()
                for writer_row in range(writer_row_start, writer_row_end):
                    (global_cell_index, cell_dof) = V_0__dof_map_writer_mapping[writer_row]
                    matrix_row_permutation[writer_row] = V_0__dof_map_reader_mapping[global_cell_index][cell_dof]
                    writer_cols, _ = writer_mat.getRow(writer_row)
                    for writer_col in writer_cols:
                        if writer_col not in matrix_col_permutation:
                            (global_cell_index, cell_dof) = V_1__dof_map_writer_mapping[writer_col]
                            matrix_col_permutation[writer_col] = V_1__dof_map_reader_mapping[global_cell_index][cell_dof]
                _permutation_storage[form_name] = (matrix_row_permutation, matrix_col_permutation)
                
        return (_permutation_storage[form_name], True)
                
    @overload(backend.Vector.Type(), (Folders.Folder, str), str, Form, str, object)
    def _permutation_load(tensor, directory, filename, form, form_name, mpi_comm):
        if form_name not in _permutation_storage:
            if not PickleIO.exists_file(directory, "." + form_name):
                return (None, False)
            else:
                V_0 = wrapping.form_argument_space(form, 0)
                V_0__dof_map_reader_mapping = wrapping.build_dof_map_reader_mapping(V_0)
                V_0__dof_map_writer_mapping = PickleIO.load_file(directory, "." + form_name)
                vector_permutation = dict() # from index at time of saving to current index
                (writer_vec, loaded) = _vector_load(directory, filename)
                if not loaded:
                    return (None, False)
                writer_row_start, writer_row_end = writer_vec.getOwnershipRange()
                for writer_row in range(writer_row_start, writer_row_end):
                    (global_cell_index, cell_dof) = V_0__dof_map_writer_mapping[writer_row]
                    vector_permutation[writer_row] = V_0__dof_map_reader_mapping[global_cell_index][cell_dof]
                _permutation_storage[form_name] = vector_permutation
                
        return (_permutation_storage[form_name], True)
        
    @overload(backend.Matrix.Type(), (Folders.Folder, str), str, object)
    def _tensor_load(tensor, directory, filename, matrix_permutation):
        (matrix_row_permutation, matrix_col_permutation) = matrix_permutation
        (writer_mat, loaded) = _matrix_load(directory, filename)
        if not loaded:
            return False
        else:
            mat = as_backend_type(tensor).mat()
            writer_row_start, writer_row_end = writer_mat.getOwnershipRange()
            for writer_row in range(writer_row_start, writer_row_end):
                row = matrix_row_permutation[writer_row]
                writer_cols, vals = writer_mat.getRow(writer_row)
                cols = list()
                for writer_col in writer_cols:
                    cols.append(matrix_col_permutation[writer_col])
                mat.setValues(row, cols, vals, addv=PETSc.InsertMode.INSERT)
            mat.assemble()
            return True
            
    @overload(backend.Vector.Type(), (Folders.Folder, str), str, object)
    def _tensor_load(tensor, directory, filename, vector_permutation):
        (writer_vec, loaded) = _vector_load(directory, filename)
        if not loaded:
            return False
        else:
            vec = as_backend_type(tensor).vec()
            writer_row_start, writer_row_end = writer_vec.getOwnershipRange()
            for writer_row in range(writer_row_start, writer_row_end):
                vec.setValues(vector_permutation[writer_row], writer_vec[writer_row], addv=PETSc.InsertMode.INSERT)
            vec.assemble()
            return True
    
    def _matrix_load(directory, filename):
        if _file_exists(directory, filename + ".dat"):
            viewer = PETSc.Viewer().createBinary(os.path.join(str(directory), filename + ".dat"), "r")
            return (PETSc.Mat().load(viewer), True)
        else:
            return (None, False)
            
    def _vector_load(directory, filename):
        if _file_exists(directory, filename + ".dat"):
            viewer = PETSc.Viewer().createBinary(os.path.join(str(directory), filename + ".dat"), "r")
            return (PETSc.Vec().load(viewer), True)
        else:
            return (None, False)
        
    def _file_exists(directory, filename):
        file_exists = False
        if is_io_process() and os.path.exists(os.path.join(str(directory), filename)):
            file_exists = True
        file_exists = is_io_process.mpi_comm.bcast(file_exists, root=is_io_process.root)
        return file_exists
    
    _permutation_storage = dict()
    
    return _basic_tensor_load

# No explicit instantiation for backend = rbnics.backends.dolfin to avoid
# circular dependencies. The concrete instatiation will be carried out in
# rbnics.backends.function.import_
