# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import overload


def basic_tensor_copy(backend, wrapping):
    @overload
    def _basic_tensor_copy(tensor: backend.Matrix.Type()):
        m = backend.Matrix(tensor.M, tensor.N)
        m[:, :] = tensor
        # Preserve auxiliary attributes related to basis functions matrix
        m._component_name_to_basis_component_index = tensor._component_name_to_basis_component_index
        m._component_name_to_basis_component_length = tensor._component_name_to_basis_component_length
        # Return
        return m

    @overload
    def _basic_tensor_copy(tensor: backend.Vector.Type()):
        v = backend.Vector(tensor.N)
        v[:] = tensor
        # Preserve auxiliary attributes related to basis functions matrix
        v._component_name_to_basis_component_index = tensor._component_name_to_basis_component_index
        v._component_name_to_basis_component_length = tensor._component_name_to_basis_component_length
        # Return
        return v

    return _basic_tensor_copy
