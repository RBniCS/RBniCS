# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.online.basic import sum as basic_sum
from rbnics.backends.online.numpy.product import ProductOutput
from rbnics.utils.decorators import backend_for


# product function to assemble truth/reduced affine expansions. To be used in combination with product,
# even though product actually carries out both the sum and the product!
@backend_for("numpy", inputs=(ProductOutput, ))
def sum(product_output):
    return basic_sum(product_output)
