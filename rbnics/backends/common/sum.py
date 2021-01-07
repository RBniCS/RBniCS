# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.backends.common.product import ProductOutput
from rbnics.utils.decorators import backend_for, list_of, overload, tuple_of
python_sum = sum


# product function to assemble truth/reduced affine expansions. To be used in combination with product,
# even though product actually carries out both the sum and the product!
@backend_for("common", inputs=((list_of(Number), ProductOutput, tuple_of(Number)), ))
def sum(args):
    return _sum(args)


@overload
def _sum(args: ProductOutput):
    return args.sum_product_return_value


@overload
def _sum(args: (list_of(Number), tuple_of(Number))):
    return python_sum(args)
