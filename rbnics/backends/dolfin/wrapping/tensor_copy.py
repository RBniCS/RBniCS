# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def tensor_copy(tensor):
    output = tensor.copy()
    # Preserve generator for I/O
    if hasattr(tensor, "generator"):
        output.generator = tensor.generator
    #
    return output
