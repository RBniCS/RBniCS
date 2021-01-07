# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.utils.decorators import backend_for, list_of
from rbnics.utils.io import Folders, TextIO


@backend_for("common", inputs=(list_of(Number), (Folders.Folder, str), str, (int, None), None))
def import_(solution, directory, filename, suffix=None, component=None):
    if suffix is not None:
        filename = filename + "_" + str(suffix)
    if TextIO.exists_file(directory, filename):
        loaded_solution = TextIO.load_file(directory, filename)
        assert len(solution) == len(loaded_solution)
        for (i, solution_i) in enumerate(loaded_solution):
            solution[i] = float(solution_i)
    else:
        raise OSError
