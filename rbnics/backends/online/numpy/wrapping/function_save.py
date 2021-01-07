# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io import NumpyIO


def function_save(fun, directory, filename, suffix=None):
    if suffix is not None:
        filename = filename + "." + str(suffix)
    NumpyIO.save_file(fun.vector(), directory, filename)
