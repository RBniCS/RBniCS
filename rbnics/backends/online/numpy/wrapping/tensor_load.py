# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io import NumpyIO


def tensor_load(tensor, directory, filename):
    if NumpyIO.exists_file(directory, filename):
        loaded = NumpyIO.load_file(directory, filename)
        assert len(loaded.shape) in (1, 2)
        if len(loaded.shape) == 1:
            tensor[:] = loaded
        elif len(loaded.shape) == 2:
            tensor[:, :] = loaded
        else:
            raise ValueError("Invalid tensor shape")
    else:
        raise OSError
