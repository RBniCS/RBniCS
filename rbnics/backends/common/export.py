# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.utils.decorators import backend_for, list_of, overload
from rbnics.utils.io import Folders, TextIO

NotImplementedType = type(NotImplemented)


@backend_for("common", inputs=((list_of(NotImplementedType), list_of(Number)), (Folders.Folder, str), str,
                               (int, None), None))
def export(solution, directory, filename, suffix=None, component=None):
    _export(solution, directory, filename, suffix, component)


# used while trying to write out scalar outputs for a problem without any output
@overload(list_of(NotImplementedType), (Folders.Folder, str), str, (int, None), None)
def _export(solution, directory, filename, suffix=None, component=None):
    pass


@overload(list_of(Number), (Folders.Folder, str), str, (int, None), None)
def _export(solution, directory, filename, suffix=None, component=None):
    if suffix is not None:
        filename = filename + "_" + str(suffix)
    TextIO.save_file(solution, directory, filename)
