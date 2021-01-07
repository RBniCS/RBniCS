# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io import NumpyIO


def function_load(fun, directory, filename, suffix=None):
    if suffix is not None:
        filename = filename + "." + str(suffix)
    file_exists = NumpyIO.exists_file(directory, filename)
    if file_exists:
        vec = NumpyIO.load_file(directory, filename)
        fun.vector()[:] = vec
    return file_exists
