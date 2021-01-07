# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
