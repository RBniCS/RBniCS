# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.online.basic import product as basic_product
from rbnics.backends.online.numpy.affine_expansion_storage import AffineExpansionStorage
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.non_affine_expansion_storage import NonAffineExpansionStorage
from rbnics.backends.online.numpy.transpose import DelayedTransposeWithArithmetic
from rbnics.backends.online.numpy.vector import Vector
from rbnics.utils.decorators import backend_for, ModuleWrapper, ThetaType

backend = ModuleWrapper(AffineExpansionStorage, Function, Matrix, NonAffineExpansionStorage, Vector)
wrapping = ModuleWrapper(DelayedTransposeWithArithmetic=DelayedTransposeWithArithmetic)
(product_base, ProductOutput) = basic_product(backend, wrapping)


# product function to assemble truth/reduced affine expansions. To be used in combination with sum,
# even though this one actually carries out both the sum and the product!
@backend_for("numpy", inputs=(ThetaType, (AffineExpansionStorage, NonAffineExpansionStorage), ThetaType + (None,)))
def product(thetas, operators, thetas2=None):
    return product_base(thetas, operators, thetas2)
