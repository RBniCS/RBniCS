# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.dolfin.product import ProductOutput
from rbnics.utils.decorators import backend_for


@backend_for("dolfin", inputs=(ProductOutput, ))
def sum(product_output):
    return product_output.sum_product_return_value
