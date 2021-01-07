# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.dolfin.wrapping.is_problem_solution import _remove_all_indices, _split_function
from rbnics.utils.cache import Cache
from rbnics.utils.decorators.store_map_from_solution_dot_to_problem import _solution_dot_to_problem_map


def is_problem_solution_dot(node):
    _prepare_solution_dot_split_storage()
    node = _remove_all_indices(node)
    return node in _solution_dot_split_to_component


def _prepare_solution_dot_split_storage():
    for solution_dot in _solution_dot_to_problem_map:
        if solution_dot not in _solution_dot_split_to_component:
            assert solution_dot not in _solution_dot_split_to_solution_dot
            _split_function(solution_dot, _solution_dot_split_to_component, _solution_dot_split_to_solution_dot)


_solution_dot_split_to_component = Cache()
_solution_dot_split_to_solution_dot = Cache()
