# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import abstract_backend


# abs function to compute maximum absolute value of an expression, matrix or vector (for EIM).
# To be used in combination with max
@abstract_backend
def abs(arg):
    pass
