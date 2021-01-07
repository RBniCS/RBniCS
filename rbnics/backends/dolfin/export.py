# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl.core.operator import Operator
from rbnics.backends.basic import export as basic_export
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import (build_dof_map_writer_mapping, form_argument_space, form_name,
                                             function_extend_or_restrict, function_from_ufl_operators,
                                             function_save, get_function_subspace, to_petsc4py)
from rbnics.backends.dolfin.wrapping.tensor_save import basic_tensor_save
from rbnics.utils.decorators import backend_for, ModuleWrapper, overload
from rbnics.utils.io import Folders

backend = ModuleWrapper(Function, Matrix, Vector)
wrapping_for_wrapping = ModuleWrapper(build_dof_map_writer_mapping, form_argument_space, to_petsc4py,
                                      form_name=form_name)
tensor_save = basic_tensor_save(backend, wrapping_for_wrapping)
wrapping = ModuleWrapper(function_extend_or_restrict, function_save, get_function_subspace, tensor_save=tensor_save)
export_base = basic_export(backend, wrapping)


# Export a solution to file
@backend_for("dolfin", inputs=((Function.Type(), Matrix.Type(), Operator, Vector.Type()), (Folders.Folder, str),
                               str, (int, None), (int, str, None)))
def export(solution, directory, filename, suffix=None, component=None):
    _export(solution, directory, filename, suffix, component)


@overload
def _export(
    solution: (
        Function.Type(),
        Matrix.Type(),
        Vector.Type()
    ),
    directory: (
        Folders.Folder,
        str
    ),
    filename: str,
    suffix: (
        int,
        None
    ) = None,
    component: (
        int,
        str,
        None
    ) = None
):
    export_base(solution, directory, filename, suffix, component)


@overload
def _export(
    solution: Operator,
    directory: (
        Folders.Folder,
        str
    ),
    filename: str,
    suffix: (
        int,
        None
    ) = None,
    component: (
        int,
        str,
        None
    ) = None
):
    export_base(function_from_ufl_operators(solution), directory, filename, suffix, component)
