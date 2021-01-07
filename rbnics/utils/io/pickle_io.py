# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pickle
import os
from rbnics.utils.mpi import parallel_io


class PickleIO(object):
    # Save a variable to file
    @staticmethod
    def save_file(content, directory, filename):
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"

        def save_file_task():
            with open(os.path.join(str(directory), filename), "wb") as outfile:
                pickle.dump(content, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        parallel_io(save_file_task)

    # Load a variable from file
    @staticmethod
    def load_file(directory, filename):
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        with open(os.path.join(str(directory), filename), "rb") as infile:
            return pickle.load(infile)

    # Check if the file exists
    @staticmethod
    def exists_file(directory, filename):
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"

        def exists_file_task():
            return os.path.exists(os.path.join(str(directory), filename))

        return parallel_io(exists_file_task)
