# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def basic_function_copy(backend, wrapping):
    def _basic_function_copy(function):
        original_vector = function.vector()
        v = backend.Vector(original_vector.N)
        v[:] = original_vector
        # Preserve auxiliary attributes related to basis functions matrix
        v._component_name_to_basis_component_index = original_vector._component_name_to_basis_component_index
        v._component_name_to_basis_component_length = original_vector._component_name_to_basis_component_length
        # Return
        return backend.Function(v)
    return _basic_function_copy
