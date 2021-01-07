# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import DictOfThetaType, overload, ThetaType
from rbnics.utils.io import ComponentNameToBasisComponentIndexDict, OnlineSizeDict


# Implementation for empty bcs
@overload(None, None, None)
def DirichletBC(bcs, component_name_to_basis_component_index=None, N=None):
    return _DirichletBC_Empty(bcs, component_name_to_basis_component_index, N)


class _DirichletBC_Empty(object):
    def __init__(self, bcs, component_name_to_basis_component_index=None, N=None):
        pass

    def apply_to_vector(self, vector, solution=None):
        pass

    def homogeneous_apply_to_vector(self, vector):
        pass

    def apply_to_matrix(self, matrix):
        pass


# Implementation for ThetaType
@overload(ThetaType, None, None)
def DirichletBC(bcs, component_name_to_basis_component_index=None, N=None):
    return _DirichletBC_ThetaType(bcs, component_name_to_basis_component_index, N)


class _DirichletBC_ThetaType(object):
    def __init__(self, bcs, component_name_to_basis_component_index=None, N=None):
        self.bcs = bcs

    def apply_to_vector(self, vector, solution=None):
        if solution is None:
            for (i, bc_i) in enumerate(self.bcs):
                vector[i] = bc_i
        else:
            for (i, bc_i) in enumerate(self.bcs):
                vector[i] = solution[i] - bc_i

    def homogeneous_apply_to_vector(self, vector):
        for (i, _) in enumerate(self.bcs):
            vector[i] = 0.

    def apply_to_matrix(self, matrix):
        for (i, _) in enumerate(self.bcs):
            matrix[i, :] = 0.
            matrix[i, i] = 1.


# Implementation for DictOfThetaType
@overload(DictOfThetaType, ComponentNameToBasisComponentIndexDict, OnlineSizeDict)
def DirichletBC(bcs, component_name_to_basis_component_index=None, N=None):
    return _DirichletBC_DictOfThetaType(bcs, component_name_to_basis_component_index, N)


class _DirichletBC_DictOfThetaType(object):
    def __init__(self, bcs, component_name_to_basis_component_index=None, N=None):
        self.bcs = bcs
        bcs_base_index = dict()  # from component name to first index
        current_bcs_base_index = 0
        for (component_name, basis_component_index) in component_name_to_basis_component_index.items():
            bcs_base_index[component_name] = current_bcs_base_index
            current_bcs_base_index += N[component_name]
        self.bcs_base_index = bcs_base_index

    def apply_to_vector(self, vector, solution=None):
        if solution is None:
            for (component_name, component_bc) in self.bcs.items():
                for (i, bc_i) in enumerate(component_bc):
                    block_i = self.bcs_base_index[component_name] + i
                    vector[block_i] = bc_i
        else:
            for (component_name, component_bc) in self.bcs.items():
                for (i, bc_i) in enumerate(component_bc):
                    block_i = self.bcs_base_index[component_name] + i
                    vector[block_i] = solution[block_i] - bc_i

    def homogeneous_apply_to_vector(self, vector):
        for (component_name, component_bc) in self.bcs.items():
            for (i, _) in enumerate(component_bc):
                block_i = self.bcs_base_index[component_name] + i
                vector[block_i] = 0.

    def apply_to_matrix(self, matrix):
        for (component_name, component_bc) in self.bcs.items():
            for (i, _) in enumerate(component_bc):
                block_i = self.bcs_base_index[component_name] + i
                matrix[block_i, :] = 0.
                matrix[block_i, block_i] = 1.
