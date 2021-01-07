# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io import OnlineSizeDict


def preserve_solution_attributes(lhs, solution, rhs):
    # We should be solving a square system
    assert lhs.M == lhs.N
    assert lhs.N == rhs.N
    # Make sure that solution preserves auxiliary attributes related to basis functions matrix
    assert (solution.vector()._component_name_to_basis_component_index is None) == (
        solution.vector()._component_name_to_basis_component_length is None)
    if solution.vector()._component_name_to_basis_component_index is None:
        solution.vector()._component_name_to_basis_component_index = lhs._component_name_to_basis_component_index[0]
        solution.vector()._component_name_to_basis_component_length = lhs._component_name_to_basis_component_length[0]
    else:
        assert lhs._component_name_to_basis_component_index[
            0] == solution.vector()._component_name_to_basis_component_index
        assert lhs._component_name_to_basis_component_length[
            0] == solution.vector()._component_name_to_basis_component_length
    # If solving a problem with one component, update solution.vector().N to be a dict
    if (solution.vector()._component_name_to_basis_component_index is not None
            and len(solution.vector()._component_name_to_basis_component_index) == 1):
        assert isinstance(solution.vector().N, (dict, int))
        if isinstance(solution.vector().N, dict):
            assert set(solution.vector()._component_name_to_basis_component_index.keys()) == set(
                solution.vector().N.keys())
        elif isinstance(solution.vector().N, int):
            for component_name in solution.vector()._component_name_to_basis_component_index:
                break
            N_int = solution.vector().N
            N = OnlineSizeDict()
            N[component_name] = N_int
            solution.vector().N = N
        else:
            raise TypeError("Invalid solution dimension")
