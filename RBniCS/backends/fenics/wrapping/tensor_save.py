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
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector
from RBniCS.backends.fenics.wrapping_utils import build_dof_map_writer_mapping, get_form_name, get_form_argument
from RBniCS.backends.fenics.wrapping.get_mpi_comm import get_mpi_comm
from RBniCS.utils.mpi import is_io_process
from RBniCS.utils.io import ExportableList

def tensor_save(tensor, directory, filename):
    mpi_comm = tensor.mpi_comm().tompi4py()
    form = tensor.generator._form
    # Write out generator
    assert hasattr(tensor, "generator")
    full_filename_generator = str(directory) + "/" + filename + ".generator"
    if is_io_process(mpi_comm):
        with open(full_filename_generator, "w") as generator_file:
            generator_file.write(get_form_name(form))
    # Write out content
    assert isinstance(tensor, (Matrix.Type(), Vector.Type()))
    if isinstance(tensor, Matrix.Type()):
        arguments_0 = get_form_argument(form, 0)
        arguments_1 = get_form_argument(form, 1)
        assert len(arguments_0) == 1
        assert len(arguments_1) == 1
        V_0 = arguments_0[0].function_space()
        V_1 = arguments_1[0].function_space()
        V_0__dof_map_writer_mapping = build_dof_map_writer_mapping(V_0)
        V_1__dof_map_writer_mapping = build_dof_map_writer_mapping(V_1)
        matrix_content = list()
        mat = as_backend_type(tensor).mat()
        row_start, row_end = mat.getOwnershipRange()
        for row in range(row_start, row_end):
            cols, vals = mat.getRow(row)
            for (col, val) in zip(cols, vals):
                if val != 0.:
                    matrix_content.append(V_0__dof_map_writer_mapping[row])
                    matrix_content.append(V_1__dof_map_writer_mapping[col])
                    matrix_content.append(val)
        gathered_matrix_content = mpi_comm.gather(matrix_content, root=is_io_process.root)
        gathered_matrix_content_flattened = ExportableList("pickle")
        if is_io_process(mpi_comm):
            for matrix_content in gathered_matrix_content:
                gathered_matrix_content_flattened.extend(matrix_content)
        gathered_matrix_content_flattened.save(directory, filename)
    elif isinstance(tensor, Vector.Type()):
        arguments_0 = get_form_argument(form, 0)
        assert len(arguments_0) == 1
        V_0 = arguments_0[0].function_space()
        V_0__dof_map_writer_mapping = build_dof_map_writer_mapping(V_0)
        vector_content = list()
        vec = as_backend_type(tensor).vec()
        row_start, row_end = vec.getOwnershipRange()
        for row in range(row_start, row_end):
            val = vec.array[row - row_start]
            if val != 0.:
                vector_content.append(V_0__dof_map_writer_mapping[row])
                vector_content.append(val)
        gathered_vector_content = mpi_comm.gather(vector_content, root=is_io_process.root)
        gathered_vector_content_flattened = ExportableList("pickle")
        if is_io_process(mpi_comm):
            for vector_content in gathered_vector_content:
                gathered_vector_content_flattened.extend(vector_content)
        gathered_vector_content_flattened.save(directory, filename)
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in tensor_save.")    
    
