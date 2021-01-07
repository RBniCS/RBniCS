# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import csv
from rbnics.utils.mpi import parallel_io


class CSVIO(object):
    # Save a variable to file
    @staticmethod
    def save_file(content, directory, filename):
        if not filename.endswith(".csv"):
            filename = filename + ".csv"

        def save_file_task():
            with open(os.path.join(str(directory), filename), "w") as outfile:
                writer = csv.writer(outfile, delimiter=";")
                writer.writerows(content)

        parallel_io(save_file_task)

    # Load a variable from file
    @staticmethod
    def load_file(directory, filename):
        if not filename.endswith(".csv"):
            filename = filename + ".csv"
        with open(os.path.join(str(directory), filename), "r") as infile:
            reader = csv.reader(infile, delimiter=";")
            return [line for line in reader]

    # Check if the file exists
    @staticmethod
    def exists_file(directory, filename):
        if not filename.endswith(".csv"):
            filename = filename + ".csv"

        def exists_file_task():
            return os.path.exists(os.path.join(str(directory), filename))

        return parallel_io(exists_file_task)
