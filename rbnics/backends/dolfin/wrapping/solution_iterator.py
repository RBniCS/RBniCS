# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl.corealg.traversal import pre_traversal


def solution_iterator(solution):
    for node in pre_traversal(solution):
        yield node
