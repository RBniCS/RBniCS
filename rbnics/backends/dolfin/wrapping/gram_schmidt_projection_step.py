# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def gram_schmidt_projection_step(new_basis, inner_product, old_basis, transpose):
    new_basis.vector().add_local(- (transpose(new_basis) * inner_product * old_basis) * old_basis.vector().get_local())
    new_basis.vector().apply("add")
    return new_basis
