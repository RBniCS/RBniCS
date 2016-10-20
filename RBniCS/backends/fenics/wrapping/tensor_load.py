# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import as_backend_type
from petsc4py import PETSc
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector
from RBniCS.backends.fenics.parametrized_tensor_factory import ParametrizedTensorFactory
from RBniCS.backends.fenics.wrapping_utils import build_dof_map_reader_mapping
from RBniCS.backends.fenics.wrapping.get_mpi_comm import get_mpi_comm
from RBniCS.backends.fenics.wrapping.tensor_copy import tensor_copy
from RBniCS.utils.mpi import is_io_process
from RBniCS.utils.io import PickleIO

def tensor_load(directory, filename, V):
    mpi_comm = get_mpi_comm(V)
    # Read in generator
    full_filename_generator = str(directory) + "/" + filename + ".generator"
    generator_string = None
    if is_io_process(mpi_comm):
        with open(full_filename_generator, "r") as generator_file:
            generator_string = generator_file.readline()
    generator_string = mpi_comm.bcast(generator_string, root=is_io_process.root)
    # Generate container based on generator
    form = ParametrizedTensorFactory._all_forms[generator_string]
    tensor = tensor_copy(ParametrizedTensorFactory._all_forms_assembled_containers[generator_string])
    tensor.zero()
    # Read in generator mpi size
    full_filename_generator_mpi_size = str(directory) + "/" + filename + ".generator_mpi_size"
    generator_mpi_size_string = None
    if is_io_process(mpi_comm):
        with open(full_filename_generator_mpi_size, "r") as generator_mpi_size_file:
            generator_mpi_size_string = generator_mpi_size_file.readline()
    generator_mpi_size_string = mpi_comm.bcast(generator_mpi_size_string, root=is_io_process.root)
    # Read in generator mapping from processor dependent indices (at the time of saving) to processor independent (global_cell_index, cell_dof) tuple
    permutation = permutation_load(V, tensor, directory, filename, form, generator_string + "_" + generator_mpi_size_string, mpi_comm)
    # Read in content
    assert isinstance(tensor, (Matrix.Type(), Vector.Type()))
    if isinstance(tensor, Matrix.Type()):
        (matrix_row_permutation, matrix_col_permutation) = permutation
        writer_mat = matrix_load(directory, filename)
        mat = as_backend_type(tensor).mat()
        writer_row_start, writer_row_end = writer_mat.getOwnershipRange()
        for writer_row in range(writer_row_start, writer_row_end):
            row = matrix_row_permutation[writer_row]
            writer_cols, vals = writer_mat.getRow(writer_row)
            cols = list()
            for writer_col in writer_cols:
                cols.append( matrix_col_permutation[writer_col] )
            mat.setValues(row, cols, vals, addv=PETSc.InsertMode.INSERT)
        mat.assemble()
    elif isinstance(tensor, Vector.Type()):
        vector_permutation = permutation
        writer_vec = vector_load(directory, filename)
        vec = as_backend_type(tensor).vec()
        writer_row_start, writer_row_end = writer_vec.getOwnershipRange()
        for writer_row in range(writer_row_start, writer_row_end):
            vec.setValues(vector_permutation[writer_row], writer_vec[writer_row], addv=PETSc.InsertMode.INSERT)
        vec.assemble()
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in tensor_load.")
    # Return
    return tensor
    
def permutation_load(V, tensor, directory, filename, form, form_name, mpi_comm):
    if not form_name in _permutation_storage:
        assert isinstance(tensor, (Matrix.Type(), Vector.Type()))
        if isinstance(tensor, Matrix.Type()):
            assert len(V) == 2
            V_0__dof_map_reader_mapping = build_dof_map_reader_mapping(V[0])
            V_1__dof_map_reader_mapping = build_dof_map_reader_mapping(V[1])
            (V_0__dof_map_writer_mapping, V_1__dof_map_writer_mapping) = PickleIO.load_file(directory, "." + form_name)
            matrix_row_permutation = dict() # from row index at time of saving to current row index
            matrix_col_permutation = dict() # from col index at time of saving to current col index
            writer_mat = matrix_load(directory, filename)
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
        elif isinstance(tensor, Vector.Type()):
            assert len(V) == 1
            V_0__dof_map_reader_mapping = build_dof_map_reader_mapping(V[0])
            V_0__dof_map_writer_mapping = PickleIO.load_file(directory, "." + form_name)
            vector_permutation = dict() # from index at time of saving to current index
            writer_vec = vector_load(directory, filename)
            writer_row_start, writer_row_end = writer_vec.getOwnershipRange()
            for writer_row in range(writer_row_start, writer_row_end):
                (global_cell_index, cell_dof) = V_0__dof_map_writer_mapping[writer_row]
                vector_permutation[writer_row] = V_0__dof_map_reader_mapping[global_cell_index][cell_dof]
            _permutation_storage[form_name] = vector_permutation
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in permutation_load.")
            
    return _permutation_storage[form_name]
    
def matrix_load(directory, filename):
    viewer = PETSc.Viewer().createBinary(str(directory) + "/" + filename + ".dat", "r")
    return PETSc.Mat().load(viewer)
    
def vector_load(directory, filename):
    viewer = PETSc.Viewer().createBinary(str(directory) + "/" + filename + ".dat", "r")
    return PETSc.Vec().load(viewer)
    
_permutation_storage = dict()

