# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import builtins
from mpi4py.MPI import COMM_WORLD


# Override the print() method to print only from process 0 of MPI_COMM_WORLD in parallel
builtin_print = builtins.print


def print(*args, **kwargs):
    if COMM_WORLD.rank == 0:
        kwargs["flush"] = True
        return builtin_print(*args, **kwargs)


builtins.print = print
