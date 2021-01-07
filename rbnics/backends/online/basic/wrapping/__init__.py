# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.online.basic.wrapping.delayed_transpose_with_arithmetic import DelayedTransposeWithArithmetic
from rbnics.backends.online.basic.wrapping.DirichletBC import DirichletBC
from rbnics.backends.online.basic.wrapping.function_to_vector import function_to_vector
from rbnics.backends.online.basic.wrapping.preserve_solution_attributes import preserve_solution_attributes
from rbnics.backends.online.basic.wrapping.slice_to_array import slice_to_array
from rbnics.backends.online.basic.wrapping.slice_to_size import slice_to_size

__all__ = [
    "DelayedTransposeWithArithmetic",
    "DirichletBC",
    "function_to_vector",
    "preserve_solution_attributes",
    "slice_to_array",
    "slice_to_size"
]
