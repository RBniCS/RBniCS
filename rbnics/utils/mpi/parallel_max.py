# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
