# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.basic.wrapping import DelayedBasisFunctionsMatrix, DelayedLinearSolver, DelayedTranspose
from rbnics.utils.decorators import backend_for


@backend_for("common", inputs=((DelayedBasisFunctionsMatrix, DelayedLinearSolver), ))
def transpose(arg):
    return DelayedTranspose(arg)
