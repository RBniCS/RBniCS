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

from mpi4py.MPI import COMM_WORLD, MAX

# Get max in parallel
def parallel_max(local_value_max, local_args=None, postprocessor=None, mpi_comm=None):
    if postprocessor is None:
        def postprocessor(value):
            return value
    if mpi_comm is None:
        mpi_comm = COMM_WORLD
    local_value_max_with_postprocessing = postprocessor(local_value_max)
    global_value_max_with_postprocessing = mpi_comm.allreduce(local_value_max_with_postprocessing, op=MAX)
    global_value_processor_argmax = -1
    if global_value_max_with_postprocessing == local_value_max_with_postprocessing:
        global_value_processor_argmax = mpi_comm.rank
    global_value_processor_argmax = mpi_comm.allreduce(global_value_processor_argmax, op=MAX)
    assert global_value_processor_argmax >= 0
    global_value_max = mpi_comm.bcast(local_value_max, root=global_value_processor_argmax)
    if local_args is not None:
        if not isinstance(local_args, tuple):
            local_args = (local_args, )
        global_args = list()
        for local_arg in local_args:
            global_args.append(
                mpi_comm.bcast(local_arg, root=global_value_processor_argmax)
            )
        return (global_value_max, tuple(global_args))
    else:
        return global_value_max
