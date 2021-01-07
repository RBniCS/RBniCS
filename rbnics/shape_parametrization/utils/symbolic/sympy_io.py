# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from sympy import python
from rbnics.shape_parametrization.utils.symbolic.sympy_exec import sympy_exec
from rbnics.utils.mpi import parallel_io


class SympyIO(object):
    # Save a variable to file
    @staticmethod
    def save_file(content, directory, filename):
        if not filename.endswith(".sym"):
            filename = filename + ".sym"

        def save_file_task():
            with open(os.path.join(str(directory), filename), "w") as outfile:
                outfile.write(python(content))

        parallel_io(save_file_task)

    # Load a variable from file
    @staticmethod
    def load_file(directory, filename):
        if not filename.endswith(".sym"):
            filename = filename + ".sym"
        with open(os.path.join(str(directory), filename), "r") as infile:
            content = infile.read()
            return sympy_exec(content, {})

    # Check if the file exists
    @staticmethod
    def exists_file(directory, filename):
        if not filename.endswith(".sym"):
            filename = filename + ".sym"

        def exists_file_task():
            return os.path.exists(os.path.join(str(directory), filename))

        return parallel_io(exists_file_task)
