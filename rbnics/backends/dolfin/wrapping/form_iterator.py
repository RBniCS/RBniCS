# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import pre_traversal


def form_iterator(form, iterator_type="nodes"):
    assert iterator_type in ("nodes", "integrals")
    if iterator_type == "nodes":
        for integral in form.integrals():
            for expression in iter_expressions(integral):
                # Note: pre_traversal algorithms guarantees that subsolutions are processed before solutions
                for node in pre_traversal(expression):
                    yield node
    elif iterator_type == "integrals":
        for integral in form.integrals():
            yield integral
    else:
        raise ValueError("Invalid iterator type")
