# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import numpy
from rbnics.utils.mpi import parallel_io


class NumpyIO(object):
    # Save a variable to file
    @staticmethod
    def save_file(content, directory, filename):
        if not filename.endswith(".npy"):
            filename = filename + ".npy"

        def save_file_task():
            numpy.save(os.path.join(str(directory), filename), content, allow_pickle=True)

        parallel_io(save_file_task)

    # Load a variable from file
    @staticmethod
    def load_file(directory, filename):
        if not filename.endswith(".npy"):
            filename = filename + ".npy"
        return numpy.load(os.path.join(str(directory), filename), allow_pickle=True)

    # Check if the file exists
    @staticmethod
    def exists_file(directory, filename):
        if not filename.endswith(".npy"):
            filename = filename + ".npy"

        def exists_file_task():
            return os.path.exists(os.path.join(str(directory), filename))

        return parallel_io(exists_file_task)
