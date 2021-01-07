# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.dolfin.wrapping.is_problem_solution_dot import (
    _solution_dot_split_to_component, _solution_dot_split_to_solution_dot)
from rbnics.backends.dolfin.wrapping.solution_identify_component import _remove_mute_indices


def solution_dot_identify_component(node):
    node = _remove_mute_indices(node)
    return _solution_dot_identify_component(node)


def _solution_dot_identify_component(node):
    assert node in _solution_dot_split_to_component
    assert node in _solution_dot_split_to_solution_dot
    return (node, _solution_dot_split_to_component[node], _solution_dot_split_to_solution_dot[node])
