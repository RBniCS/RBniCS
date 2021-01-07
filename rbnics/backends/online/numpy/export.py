# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.basic import export as basic_export
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import function_save, tensor_save
from rbnics.utils.decorators import backend_for, ModuleWrapper
from rbnics.utils.io import Folders

backend = ModuleWrapper(Function, Matrix, Vector)
wrapping = ModuleWrapper(function_save, tensor_save)
export_base = basic_export(backend, wrapping)


# Export a solution to file
@backend_for("numpy", inputs=((Function.Type(), Matrix.Type(), Vector.Type()), (Folders.Folder, str),
                              str, (int, None), (int, str, None)))
def export(solution, directory, filename, suffix=None, component=None):
    export_base(solution, directory, filename, suffix, component)
