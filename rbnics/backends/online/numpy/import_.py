# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.basic import import_ as basic_import_
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import function_load, tensor_load
from rbnics.utils.decorators import backend_for, ModuleWrapper
from rbnics.utils.io import Folders

backend = ModuleWrapper(Function, Matrix, Vector)
wrapping = ModuleWrapper(function_load, tensor_load)
import_base = basic_import_(backend, wrapping)


# Import a solution from file
@backend_for("numpy", inputs=((Function.Type(), Matrix.Type(), Vector.Type()), (Folders.Folder, str),
                              str, (int, None), (int, str, None)))
def import_(solution, directory, filename, suffix=None, component=None):
    import_base(solution, directory, filename, suffix, component)
