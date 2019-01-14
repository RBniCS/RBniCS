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

import os
from petsc4py import PETSc
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import overload
from rbnics.utils.io import Folders, PickleIO
from rbnics.utils.mpi import parallel_io

def basic_tensor_load(backend, wrapping):
    def _basic_tensor_load(tensor, directory, filename):
        mpi_comm = tensor.mpi_comm()
        form = tensor.generator._form
        # Read in generator
        full_filename_generator = os.path.join(str(directory), filename + ".generator")
        def load_generator():
            if os.path.exists(full_filename_generator):
                with open(full_filename_generator, "r") as generator_file:
                    return generator_file.readline()
            else:
                raise OSError
        generator_string = parallel_io(load_generator, mpi_comm)
        # Read in generator mpi size
        full_filename_generator_mpi_size = os.path.join(str(directory), filename + ".generator_mpi_size")
        def load_generator_mpi_size():
            if os.path.exists(full_filename_generator_mpi_size):
                with open(full_filename_generator_mpi_size, "r") as generator_mpi_size_file:
                    return generator_mpi_size_file.readline()
            else:
                raise OSError
        generator_mpi_size_string = parallel_io(load_generator_mpi_size, mpi_comm)
        # Read in generator mapping from processor dependent indices (at the time of saving) to processor independent (global_cell_index, cell_dof) tuple
        permutation = _permutation_load(tensor, directory, filename, form, generator_string + "_" + generator_mpi_size_string, mpi_comm)
        _tensor_load(tensor, directory, filename, permutation, mpi_comm)
        
    @overload(backend.Matrix.Type(), (Folders.Folder, str), str, object, str, object)
    def _permutation_load(tensor, directory, filename, form, form_name, mpi_comm):
        if form_name not in _permutation_storage:
            if not PickleIO.exists_file(directory, "." + form_name):
                raise OSError
            else:
                V_0 = wrapping.form_argument_space(form, 0)
                V_1 = wrapping.form_argument_space(form, 1)
                V_0__dof_map_reader_mapping = wrapping.build_dof_map_reader_mapping(V_0)
                V_1__dof_map_reader_mapping = wrapping.build_dof_map_reader_mapping(V_1)
                (V_0__dof_map_writer_mapping, V_1__dof_map_writer_mapping) = PickleIO.load_file(directory, "." + form_name)
                matrix_row_permutation = dict() # from row index at time of saving to current row index
                matrix_col_permutation = dict() # from col index at time of saving to current col index
                writer_mat = _matrix_load(directory, filename, mpi_comm)
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
                
        return _permutation_storage[form_name]
                
    @overload(backend.Vector.Type(), (Folders.Folder, str), str, object, str, object)
    def _permutation_load(tensor, directory, filename, form, form_name, mpi_comm):
        if form_name not in _permutation_storage:
            if not PickleIO.exists_file(directory, "." + form_name):
                raise OSError
            else:
                V_0 = wrapping.form_argument_space(form, 0)
                V_0__dof_map_reader_mapping = wrapping.build_dof_map_reader_mapping(V_0)
                V_0__dof_map_writer_mapping = PickleIO.load_file(directory, "." + form_name)
                vector_permutation = dict() # from index at time of saving to current index
                writer_vec = _vector_load(directory, filename, mpi_comm)
                writer_row_start, writer_row_end = writer_vec.getOwnershipRange()
                for writer_row in range(writer_row_start, writer_row_end):
                    (global_cell_index, cell_dof) = V_0__dof_map_writer_mapping[writer_row]
                    vector_permutation[writer_row] = V_0__dof_map_reader_mapping[global_cell_index][cell_dof]
                _permutation_storage[form_name] = vector_permutation
                
        return _permutation_storage[form_name]
        
    @overload(backend.Matrix.Type(), (Folders.Folder, str), str, object, object)
    def _tensor_load(tensor, directory, filename, matrix_permutation, mpi_comm):
        (matrix_row_permutation, matrix_col_permutation) = matrix_permutation
        writer_mat = _matrix_load(directory, filename, mpi_comm)
        mat = wrapping.to_petsc4py(tensor)
        writer_row_start, writer_row_end = writer_mat.getOwnershipRange()
        for writer_row in range(writer_row_start, writer_row_end):
            row = matrix_row_permutation[writer_row]
            writer_cols, writer_vals = writer_mat.getRow(writer_row)
            cols = list()
            vals = list()
            for (writer_col, writer_val) in zip(writer_cols, writer_vals):
                if writer_val != 0.:
                    cols.append(matrix_col_permutation[writer_col])
                    vals.append(writer_val)
            if len(cols) > 0:
                mat.setValues(row, cols, vals, addv=PETSc.InsertMode.INSERT)
        mat.assemble()
        
    @overload(backend.Vector.Type(), (Folders.Folder, str), str, object, object)
    def _tensor_load(tensor, directory, filename, vector_permutation, mpi_comm):
        writer_vec = _vector_load(directory, filename, mpi_comm)
        vec = wrapping.to_petsc4py(tensor)
        writer_row_start, writer_row_end = writer_vec.getOwnershipRange()
        for writer_row in range(writer_row_start, writer_row_end):
            vec.setValues(vector_permutation[writer_row], writer_vec[writer_row], addv=PETSc.InsertMode.INSERT)
        vec.assemble()
    
    def _matrix_load(directory, filename, mpi_comm):
        if _file_exists(directory, filename + ".dat", mpi_comm):
            viewer = PETSc.Viewer().createBinary(os.path.join(str(directory), filename + ".dat"), "r", mpi_comm)
            return PETSc.Mat().load(viewer)
        else:
            raise OSError
            
    def _vector_load(directory, filename, mpi_comm):
        if _file_exists(directory, filename + ".dat", mpi_comm):
            viewer = PETSc.Viewer().createBinary(os.path.join(str(directory), filename + ".dat"), "r", mpi_comm)
            return PETSc.Vec().load(viewer)
        else:
            raise OSError
        
    def _file_exists(directory, filename, mpi_comm):
        def file_exists_task():
            return os.path.exists(os.path.join(str(directory), filename))
        return parallel_io(file_exists_task, mpi_comm)
    
    _permutation_storage = Cache()
    
    return _basic_tensor_load

# No explicit instantiation for backend = rbnics.backends.dolfin to avoid
# circular dependencies. The concrete instatiation will be carried out in
# rbnics.backends.dolfin.import_
