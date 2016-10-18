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
from mpi4py.MPI import Op
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector
from RBniCS.backends.fenics.wrapping_utils import build_dof_map_writer_mapping, get_form_name, get_form_argument
from RBniCS.backends.fenics.wrapping.get_mpi_comm import get_mpi_comm
from RBniCS.utils.mpi import is_io_process
from RBniCS.utils.io import PickleIO

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
        matrix_content = dict()
        mat = as_backend_type(tensor).mat()
        row_start, row_end = mat.getOwnershipRange()
        for row in range(row_start, row_end):
            cols, vals = mat.getRow(row)
            for (col, val) in zip(cols, vals):
                if val != 0.:
                    if V_0__dof_map_writer_mapping[row] not in matrix_content:
                        matrix_content[V_0__dof_map_writer_mapping[row]] = dict()
                    matrix_content[V_0__dof_map_writer_mapping[row]][V_1__dof_map_writer_mapping[col]] = val
        gathered_matrix_content = mpi_comm.reduce(matrix_content, root=is_io_process.root, op=dict_update_op)
        PickleIO.save_file(gathered_matrix_content, directory, filename)
    elif isinstance(tensor, Vector.Type()):
        arguments_0 = get_form_argument(form, 0)
        assert len(arguments_0) == 1
        V_0 = arguments_0[0].function_space()
        V_0__dof_map_writer_mapping = build_dof_map_writer_mapping(V_0)
        vector_content = dict()
        vec = as_backend_type(tensor).vec()
        row_start, row_end = vec.getOwnershipRange()
        for row in range(row_start, row_end):
            val = vec.array[row - row_start]
            if val != 0.:
                vector_content[V_0__dof_map_writer_mapping[row]] = val
        gathered_vector_content = mpi_comm.reduce(vector_content, root=is_io_process.root, op=dict_update_op)
        PickleIO.save_file(gathered_vector_content, directory, filename)
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in tensor_save.")    
    
def dict_update(dict1, dict2, datatype):
    dict1.update(dict2)
    return dict1

dict_update_op = Op.Create(dict_update, commute=True)

