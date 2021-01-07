# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io import NumpyIO


def tensor_save(tensor, directory, filename):
    NumpyIO.save_file(tensor, directory, filename)
