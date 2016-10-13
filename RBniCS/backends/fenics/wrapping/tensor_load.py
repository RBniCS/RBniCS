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
from RBniCS.backends.fenics.projected_parametrized_tensor import ProjectedParametrizedTensor
from RBniCS.backends.fenics.wrapping_utils import build_dof_map_reader_mapping, get_form_argument
from RBniCS.backends.fenics.wrapping.get_mpi_comm import get_mpi_comm
from RBniCS.backends.fenics.wrapping.tensor_copy import tensor_copy
from RBniCS.utils.mpi import is_io_process
from RBniCS.utils.io import ExportableList

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
    form = ProjectedParametrizedTensor._all_forms[generator_string]
    tensor = tensor_copy(ProjectedParametrizedTensor._all_forms_assembled_containers[generator_string])
    tensor.zero()
    # Read in content
    assert isinstance(tensor, (Matrix.Type(), Vector.Type()))
    if isinstance(tensor, Matrix.Type()):
        arguments_0 = get_form_argument(form, 0)
        arguments_1 = get_form_argument(form, 1)
        assert len(arguments_0) == 1
        assert len(arguments_1) == 1
        V_0 = arguments_0[0].function_space()
        V_1 = arguments_1[0].function_space()
        V_0__dof_map_reader_mapping = build_dof_map_reader_mapping(V_0)
        V_1__dof_map_reader_mapping = build_dof_map_reader_mapping(V_1)
        matrix_content = ExportableList("pickle")
        matrix_content.load(directory, filename)
        mat = as_backend_type(tensor).mat()
        row_start, row_end = mat.getOwnershipRange()
        matrix_content_iterator = 0
        prev_row = -1
        all_cols = list()
        all_vals = list()
        while matrix_content_iterator < len(matrix_content):
            (global_cell_index, cell_dof) = (matrix_content[matrix_content_iterator][0], matrix_content[matrix_content_iterator][1])
            row = V_0__dof_map_reader_mapping[global_cell_index][cell_dof]
            matrix_content_iterator += 1
            if row >= row_start and row < row_end:
                (global_cell_index, cell_dof) = (matrix_content[matrix_content_iterator][0], matrix_content[matrix_content_iterator][1])
                col = V_1__dof_map_reader_mapping[global_cell_index][cell_dof]
                matrix_content_iterator += 1
                val = matrix_content[matrix_content_iterator]
                matrix_content_iterator += 1
                if row != prev_row and prev_row != -1:
                    assert len(all_cols) == len(all_vals)
                    mat.setValues(prev_row, all_cols, all_vals, addv=PETSc.InsertMode.INSERT)
                    all_cols = list()
                    all_vals = list()
                prev_row = row
                all_cols.append(col)
                all_vals.append(val)
            else:
                matrix_content_iterator += 2
        # Do not forget the last read row!
        assert len(all_cols) == len(all_vals)
        mat.setValues(prev_row, all_cols, all_vals, addv=PETSc.InsertMode.INSERT)
        mat.assemble()
    elif isinstance(tensor, Vector.Type()):
        arguments_0 = get_form_argument(form, 0)
        assert len(arguments_0) == 1
        V_0 = arguments_0[0].function_space()
        V_0__dof_map_reader_mapping = build_dof_map_reader_mapping(V_0)
        vector_content = ExportableList("pickle")
        vector_content.load(directory, filename)
        vec = as_backend_type(tensor).vec()
        row_start, row_end = vec.getOwnershipRange()
        vector_content_iterator = 0
        while vector_content_iterator < len(vector_content):
            (global_cell_index, cell_dof) = (vector_content[vector_content_iterator][0], vector_content[vector_content_iterator][1])
            row = V_0__dof_map_reader_mapping[global_cell_index][cell_dof]
            vector_content_iterator += 1
            if row >= row_start and row < row_end:
                val = vector_content[vector_content_iterator]
                vector_content_iterator += 1
                vec.setValuesLocal(row - row_start, val, addv=PETSc.InsertMode.INSERT)
            else:
                vector_content_iterator += 1
        vec.assemble()
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in tensor_load.")
    # Return
    return tensor
    
