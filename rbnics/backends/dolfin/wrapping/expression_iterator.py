# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import pre_traversal


def expression_iterator(expression):
    for subexpression in iter_expressions(expression):
        # Note: pre_traversal algorithms guarantees that subsolutions are processed before solutions
        for node in pre_traversal(subexpression):
            yield node
