# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import abstract_backend


# max function to compute maximum absolute value of an expression, matrix or vector (for EIM).
# To be used in combination with abs
@abstract_backend
def max(abs_output):
    pass
