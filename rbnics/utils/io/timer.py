# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from timeit import default_timer as python_timer
from mpi4py import MPI
from mpi4py.MPI import MAX, SUM


class Timer(object):
    def __init__(self, mode):
        assert mode in ("serial", "parallel")
        self._mode = mode
        self._start = None
        self._comm = MPI.COMM_WORLD

    def start(self):
        self._start = python_timer()

    def stop(self):
        elapsed = python_timer() - self._start
        self._start = None
        if self._mode == "serial":
            return self._comm.allreduce(elapsed, op=MAX)
        elif self._mode == "parallel":
            return self._comm.allreduce(elapsed, op=SUM)
        else:
            raise ValueError("Invalid mode for timer")
