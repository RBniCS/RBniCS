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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from petsc4py import PETSc
from dolfin import as_backend_type
from mpi4py.MPI import Op
import RBniCS.backends # avoid circular imports when importing fenics backend
from RBniCS.backends.fenics.wrapping.dofs_parallel_io_helpers import build_dof_map_writer_mapping
from RBniCS.backends.fenics.wrapping.get_form_name import get_form_name
from RBniCS.backends.fenics.wrapping.get_form_argument import get_form_argument
from RBniCS.utils.mpi import is_io_process
from RBniCS.utils.io import PickleIO

def tensor_save(tensor, directory, filename):
    mpi_comm = tensor.mpi_comm().tompi4py()
    form = tensor.generator._form
    # Write out generator
    assert hasattr(tensor, "generator")
    full_filename_generator = str(directory) + "/" + filename + ".generator"
    form_name = get_form_name(form)
    if is_io_process(mpi_comm):
        with open(full_filename_generator, "w") as generator_file:
            generator_file.write(form_name)
    # Write out generator mpi size
    full_filename_generator_mpi_size = str(directory) + "/" + filename + ".generator_mpi_size"
    if is_io_process(mpi_comm):
        with open(full_filename_generator_mpi_size, "w") as generator_mpi_size_file:
            generator_mpi_size_file.write(str(mpi_comm.size))
    # Write out generator mapping from processor dependent indices to processor independent (global_cell_index, cell_dof) tuple
    permutation_save(tensor, directory, form, form_name + "_" + str(mpi_comm.size), mpi_comm)
    # Write out content
    assert isinstance(tensor, (RBniCS.backends.fenics.Matrix.Type(), RBniCS.backends.fenics.Vector.Type()))
    if isinstance(tensor, RBniCS.backends.fenics.Matrix.Type()):
        matrix_save(tensor, directory, filename)
    elif isinstance(tensor, RBniCS.backends.fenics.Vector.Type()):
        vector_save(tensor, directory, filename)
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in tensor_save.")
        
def permutation_save(tensor, directory, form, form_name, mpi_comm):
    if not PickleIO.exists_file(directory, "." + form_name):
        assert isinstance(tensor, (RBniCS.backends.fenics.Matrix.Type(), RBniCS.backends.fenics.Vector.Type()))
        if isinstance(tensor, RBniCS.backends.fenics.Matrix.Type()):
            V_0 = get_form_argument(form, 0).function_space()
            V_1 = get_form_argument(form, 1).function_space()
            V_0__dof_map_writer_mapping = build_dof_map_writer_mapping(V_0)
            V_1__dof_map_writer_mapping = build_dof_map_writer_mapping(V_1)
            matrix_row_mapping = dict() # from processor dependent row indices to processor independent tuple
            matrix_col_mapping = dict() # from processor dependent col indices to processor independent tuple
            mat = as_backend_type(tensor).mat()
            row_start, row_end = mat.getOwnershipRange()
            for row in range(row_start, row_end):
                matrix_row_mapping[row] = V_0__dof_map_writer_mapping[row]
                cols, _ = mat.getRow(row)
                for col in cols:
                    if col not in matrix_col_mapping:
                        matrix_col_mapping[col] = V_1__dof_map_writer_mapping[col]
            gathered_matrix_row_mapping = mpi_comm.reduce(matrix_row_mapping, root=is_io_process.root, op=dict_update_op)
            gathered_matrix_col_mapping = mpi_comm.reduce(matrix_col_mapping, root=is_io_process.root, op=dict_update_op)
            gathered_matrix_mapping = (gathered_matrix_row_mapping, gathered_matrix_col_mapping)
            PickleIO.save_file(gathered_matrix_mapping, directory, "." + form_name)
        elif isinstance(tensor, RBniCS.backends.fenics.Vector.Type()):
            V_0 = get_form_argument(form, 0).function_space()
            V_0__dof_map_writer_mapping = build_dof_map_writer_mapping(V_0)
            vector_mapping = dict() # from processor dependent indices to processor independent tuple
            vec = as_backend_type(tensor).vec()
            row_start, row_end = vec.getOwnershipRange()
            for row in range(row_start, row_end):
                vector_mapping[row] = V_0__dof_map_writer_mapping[row]
            gathered_vector_mapping = mpi_comm.reduce(vector_mapping, root=is_io_process.root, op=dict_update_op)
            PickleIO.save_file(gathered_vector_mapping, directory, "." + form_name)
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in permutation_save.")
            
def matrix_save(tensor, directory, filename):
    mat = as_backend_type(tensor).mat()
    viewer = PETSc.Viewer().createBinary(str(directory) + "/" + filename + ".dat", "w")
    viewer.view(mat)
    
def vector_save(tensor, directory, filename):
    vec = as_backend_type(tensor).vec()
    viewer = PETSc.Viewer().createBinary(str(directory) + "/" + filename + ".dat", "w")
    viewer.view(vec)
    
def dict_update(dict1, dict2, datatype):
    dict1.update(dict2)
    return dict1

dict_update_op = Op.Create(dict_update, commute=True)

