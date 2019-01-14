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

import sys
from mpi4py.MPI import COMM_WORLD

def parallel_io(lambda_function, mpi_comm=None):
    if mpi_comm is None:
        mpi_comm = COMM_WORLD
    return_value = None
    error_raised = False
    error_type = None
    error_instance_args = None
    if mpi_comm.rank == 0:
        try:
            return_value = lambda_function()
        except Exception:
            error_raised = True
            error_type, error_instance, _ = sys.exc_info()
            error_instance_args = error_instance.args
    error_raised = mpi_comm.bcast(error_raised, root=0)
    if not error_raised:
        return_value = mpi_comm.bcast(return_value, root=0)
        return return_value
    else:
        error_type = mpi_comm.bcast(error_type, root=0)
        error_instance_args = mpi_comm.bcast(error_instance_args, root=0)
        raise error_type(*error_instance_args)
