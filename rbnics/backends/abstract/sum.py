# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import abstract_backend


# sum function to assemble truth/reduced affine expansions. To be used in combination with the product method.
@abstract_backend
def sum(product_output):
    pass
