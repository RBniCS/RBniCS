# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl import MixedElement, TensorElement, VectorElement
from dolfin import assign, Function, has_hdf5, has_hdf5_parallel
from rbnics.backends.dolfin.wrapping.function_save import _all_solution_files, SolutionFileXDMF, SolutionFileXML
from rbnics.backends.dolfin.wrapping.get_function_subspace import get_function_subspace


def function_load(fun, directory, filename, suffix=None):
    fun_V = fun.function_space()
    if hasattr(fun_V, "_index_to_components") and len(fun_V._index_to_components) > 1:
        for (index, components) in fun_V._index_to_components.items():
            sub_fun_V = get_function_subspace(fun_V, components)
            sub_fun = Function(sub_fun_V)
            _read_from_file(sub_fun, directory, filename, suffix, components)
            assign(fun.sub(index), sub_fun)
    else:
        _read_from_file(fun, directory, filename, suffix)


def _read_from_file(fun, directory, filename, suffix, components=None):
    if components is not None:
        filename = filename + "_component_" + "".join(components)
        function_name = "function_" + "".join(components)
    else:
        function_name = "function"
    fun_V_element = fun.function_space().ufl_element()
    if isinstance(fun_V_element, MixedElement) and not isinstance(fun_V_element, (TensorElement, VectorElement)):
        funs = fun.split(deepcopy=True)
        for (i, fun_i) in enumerate(funs):
            if components is not None:
                filename_i = filename + "_subcomponent_" + str(i)
            else:
                filename_i = filename + "_component_" + str(i)
            _read_from_file(fun_i, directory, filename_i, suffix, None)
            assign(fun.sub(i), fun_i)
    else:
        if fun_V_element.family() == "Real":
            SolutionFile = SolutionFileXML
        else:
            if has_hdf5() and has_hdf5_parallel():
                SolutionFile = SolutionFileXDMF
            else:
                SolutionFile = SolutionFileXML
        if suffix is not None:
            if suffix == 0:
                # Remove from storage and re-create
                try:
                    del _all_solution_files[(directory, filename)]
                except KeyError:
                    pass
                _all_solution_files[(directory, filename)] = SolutionFile(directory, filename)
            file_ = _all_solution_files[(directory, filename)]
            file_.read(fun, function_name, suffix)
        else:
            file_ = SolutionFile(directory, filename)
            file_.read(fun, function_name, 0)
