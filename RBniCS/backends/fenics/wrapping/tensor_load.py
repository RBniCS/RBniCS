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

from petsc4py import PETSc
from dolfin import PETScMatrix, PETScVector

def tensor_load(directory, filename, V_unused):
    full_filename_type = str(directory) + "/" + filename + ".type"
    mpi_comm = tensor.mpi_comm()
    type_string = None
    if is_io_process(mpi_comm):
        with open(full_filename_type, "r") as type_file:
            type_string = type_file.readline()
    type_string = mpi_comm.bcast(type_string, root=is_io_process.root)
    full_filename_content = str(directory) + "/" + filename + ".dat"
    viewer = PETSc.Viewer().createBinary(full_filename_content, "r")
    assert type_string in ("matrix", "vector")
    if type_string == "matrix":
        data = PETSc.Mat().load(viewer)
        return PETScMatrix(data)
    elif type_string == "vector":
        data = PETSc.Vec().load(viewer)
        return PETScVector(data)
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in tensor_load.")

